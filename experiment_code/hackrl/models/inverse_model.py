"""Adapted from Chaos Dwarf in Nethack Challenge Starter Kit:
https://github.com/Miffyli/nle-sample-factory-baseline

MIT License

Copyright (c) 2021 Anssi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torchvision.models as models
from nle import nethack
from torch import nn
from torch.nn import functional as F

from hackrl.models.chaotic_dwarf import ScreenEncoder


class TopLineEncoder(nn.Module):
    def __init__(self):
        super(TopLineEncoder, self).__init__()
        self.hidden_dim = 128
        self.i_dim = nethack.NLE_TERM_CO * 256

        self.msg_fwd = nn.Sequential(
            nn.Linear(self.i_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = (
            F.one_hot((message).long(), 256).reshape(-1, self.i_dim).float()
        )
        return self.msg_fwd(message_normed)


class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.train()
        self.hidden_dim = 512
        self.fwd = nn.Sequential(
            nn.Linear(1000, self.hidden_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, screen_image):
        return self.fwd(self.resnet(screen_image / 255.0))


class BigCharEncoder(nn.Module):
    def __init__(self):
        super(BigCharEncoder, self).__init__()
        self.hidden_dim = 128
        self.bidirectional = True
        self.core_dim = 128
        self.core_out = self.core_dim * 2 if self.bidirectional else self.core_dim
        self.chars = nethack.NLE_TERM_CO
        self.i_dim = 256

        self.msg_encode = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
        )
        self.core_decode = nn.Sequential(
            nn.Linear(self.core_out, self.hidden_dim),
            nn.ELU(inplace=True),
        )

        self.core = nn.GRU(
            128,
            self.core_dim,
            num_layers=1,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, message):

        B, C = message.shape
        x = F.one_hot(message.long(), 256).float()
        x = self.msg_encode(x)
        output, state = self.core(x)  # [B, C, core], [2, B, C]

        state = state.transpose(0, 1).reshape(B, 1, self.core_out)  # [batch, 1, core]
        weights = F.softmax(torch.sum(state * output, dim=2), dim=1).view(
            B, C, 1
        )  # [batch, col, 1]
        x = torch.sum(output * weights, dim=1)  # [ batch,  core]
        x = self.core_decode(x)
        return x


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


class InverseModel(nn.Module):
    def __init__(self, h_dim, action_space, use_difference_vector=False):
        super(InverseModel, self).__init__()
        self.h_dim = h_dim
        self.use_difference_vector = use_difference_vector
        if not use_difference_vector:
            self.h_dim *= 2
        self.action_space = action_space

        self.fwd_model = nn.Sequential(
            nn.Linear(self.h_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, action_space),
        )

    def forward(self, obs):
        T, B, *_ = obs.shape
        if self.use_difference_vector:
            x = obs[1:] - obs[:-1]
        else:
            x = torch.cat([obs[:-1], obs[1:]], dim=-1)
        pred_a = self.fwd_model(x)
        off_by_one = torch.ones((1, B, self.action_space), device=x.device) * -1
        return torch.cat([pred_a, off_by_one], dim=0)


class BigInverseOnlyModel(nn.Module):
    def __init__(self, shape, action_space, flags, device):
        super(BigInverseOnlyModel, self).__init__()

        self.flags = flags
        self.num_actions = len(action_space)

        self.use_inverse_model = flags.use_inverse_model
        self.use_tty_only = flags.use_tty_only
        self.use_prev_action = flags.use_prev_action

        self.topline_encoder = TopLineEncoder()

        pixel_size = flags.pixel_size
        if flags.crop_dim == 0:
            screen_shape = (24 * pixel_size, 80 * pixel_size)
        else:
            screen_shape = (flags.crop_dim * pixel_size, flags.crop_dim * pixel_size)

        if flags.use_resnet:
            self.screen_encoder = torch.jit.script(ResnetEncoder())
        else:
            self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )

        self.hidden_dim = 512
        self.inverse_model = InverseModel(
            self.h_dim, self.num_actions, self.flags.use_difference_vector
        )
        self.version = 0

    def initial_state(self, batch_size=1):
        return tuple(torch.zeros(1, batch_size, 1) for _ in range(2))

    def forward(self, inputs, core_state):
        T, B, C, H, W = inputs["screen_image"].shape

        topline = inputs["tty_chars"][..., 0, :]

        st = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .float(memory_format=torch.contiguous_format)
                .view(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            st.append(
                torch.nn.functional.one_hot(
                    inputs["prev_action"], self.num_actions
                ).view(T * B, -1)
            )

        st = torch.cat(st, dim=1)

        core_input = st.view(T, B, -1)
        inverse_action_logits = self.inverse_model(core_input)

        policy_logits = torch.ones((T * B, self.num_actions))
        baseline = torch.zeros((T * B))
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        version = torch.ones_like(action) * self.version

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            version=version,
            inverse_action_logits=inverse_action_logits,
            encoded_state=core_input,
        )
        output = {k: v.to(self.flags.device) for k, v in output.items()}

        return (output, core_state)
