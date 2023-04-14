from functools import partial

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from hackrl.core import nest
from hackrl.models.trajectory_gpt2 import GPT2Model
from hackrl.models.chaotic_dwarf import ChaoticDwarvenGPT5


class DecisionTransformer(ChaoticDwarvenGPT5):
    """This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)."""

    def __init__(
        self,
        shape, 
        action_space, 
        flags, 
        device, 
    ):
        super().__init__(shape, action_space, flags, device)

        self.max_length = flags.ttyrec_unroll_length
        self.use_returns = flags.use_returns
        self.use_actions = flags.use_actions
        self.scale = flags.scale

        self.n = 1 + self.use_actions * 1 + self.use_returns * 1

        if self.use_returns:
            self.h_dim += 1

        self.embed_input = nn.Linear(self.h_dim, self.hidden_dim)

        kwargs = dict(
            n_layer=flags.n_layer,
            n_head=flags.n_head,
            n_inner=4 * self.hidden_dim,
            activation_function=flags.activation_function,
            resid_pdrop=flags.dropout,
            attn_pdrop=flags.dropout,
        )

        config = transformers.GPT2Config(
            vocab_size=1, n_embd=self.hidden_dim, **kwargs  # doesn't matter -- we don't use the vocab
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.core = GPT2Model(config)
        self.core.hidden_size = self.hidden_dim
        self.core.num_layers = flags.n_layer

        # self.embed_timestep = nn.Embedding(flags.env.max_episode_steps, self.hidden_dim)
        self.embed_timestep = nn.Linear(1, self.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hidden_dim)


    def initial_state(self, batch_size=1):
        return dict(
            blstats=torch.zeros(self.max_length, batch_size, 27).to(torch.long),
            done=torch.zeros(self.max_length, batch_size).to(torch.bool),
            message=torch.zeros(self.max_length, batch_size, 256).to(torch.uint8),
            reward=torch.zeros(self.max_length, batch_size),
            screen_image=torch.zeros(self.max_length, batch_size, 3, 108, 108).to(torch.uint8),
            tty_chars=torch.zeros(self.max_length, batch_size, 24, 80).to(torch.uint8),
            tty_colors=torch.zeros(self.max_length, batch_size, 24, 80).to(torch.int8),
            tty_cursor=torch.zeros(self.max_length, batch_size, 2).to(torch.uint8),
            prev_action=torch.zeros(self.max_length, batch_size).to(torch.long),
            timesteps=torch.zeros(self.max_length, batch_size),
            max_scores=torch.zeros(self.max_length, batch_size).to(torch.long),
            mask=torch.zeros(self.max_length, batch_size).to(torch.bool),
        )

    def forward(self, inputs, core_state):
        org_inputs = inputs
        OT = org_inputs["screen_image"].shape[0]

        core_state = dict(sorted({key: value for key, value in core_state.items() if key in inputs.keys()}.items()))
        inputs = dict(sorted({key: value for key, value in inputs.items() if key in core_state.keys()}.items()))
        inputs = nest.map_many(partial(torch.cat, dim=0), *[core_state, inputs])
        inputs = nest.map(lambda x: x[-self.max_length :], inputs)

        T, B, C, H, W = inputs["screen_image"].shape

        if self.use_tty_only:
            topline = inputs["tty_chars"][..., 0, :]
            bottom_line = inputs["tty_chars"][..., -2:, :]
        else:
            topline = inputs["message"]
            bottom_line = inputs["blstats"]
            assert False, "TODO: topline shape transpose "

        topline = topline.permute(1, 0, 2)
        bottom_line = bottom_line.permute(1, 0, 2, 3)

        st = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).reshape(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).reshape(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"].permute(1, 0, 2, 3, 4)
                .float(memory_format=torch.contiguous_format)
                .reshape(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            st.append(
                torch.nn.functional.one_hot(
                    inputs["prev_action"].T, self.num_actions
                ).view(T * B, -1)
            )
        if self.use_returns:
            st.append(inputs["max_scores"].T.reshape(T * B, -1) / self.scale)

        st = torch.cat(st, dim=1)
        core_input = st.view(B, T, -1)
        inputs_embeds = self.embed_input(core_input) 

        timesteps = inputs["timesteps"].permute(1, 0).unsqueeze(-1) / self.scale
        time_embeddings = self.embed_timestep(timesteps)
        inputs_embeds = inputs_embeds + time_embeddings

        inputs_embeds = self.embed_ln(inputs_embeds)

        attention_mask = inputs["mask"].T
        causal_mask = torch.tril(torch.ones((B, T, T), dtype=torch.uint8)).view(B, 1, T, T).to(attention_mask.device)
        if inputs["done"].any():
            # # for breakpoint
            # if "actions_converted" in org_inputs:
            #     print("breakpoint")

            # modify causal mask, to prevent attending to states between games
            mask = torch.ones_like(causal_mask)
            xs, ys = torch.where(inputs["done"].T)
            for x, y in zip(xs, ys):
                mask[x] = 0 
                mask[x, :, y:, y:] = 1 
                mask[x, :, :y, :y] = 1 
                
                # reset state if episode finished
                init_state = self.initial_state()
                init_state = nest.map(torch.squeeze, init_state)
                nest.slice(inputs, (slice(None), x), init_state)
            causal_mask *= mask

        core_output = self.core(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        x = core_output["last_hidden_state"]

        # we only want to predict the same number of actions as seq len of org_inputs
        x = x[:, -OT:]

        # -- [B' x A]
        policy_logits = self.policy(x)

        # -- [B' x 1]
        baseline = self.baseline(x).squeeze(-1)

        action = torch.multinomial(F.softmax(policy_logits.view(B * OT, -1), dim=1), num_samples=1)

        policy_logits = policy_logits.permute(1, 0, 2)
        baseline = baseline.permute(1, 0)
        action = action.view(B, OT).permute(1, 0)

        version = torch.ones_like(action) * self.version

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            version=version,
        )

        if self.use_inverse_model:
            inverse_action_logits = self.inverse_model(core_input)
            output["encoded_state"] = core_input
            output["inverse_action_logits"] = inverse_action_logits
        return (output, inputs)


    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.max_length - states.shape[1], self.state_dim), device=states.device
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds[0, -1]