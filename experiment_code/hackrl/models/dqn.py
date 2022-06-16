import math
import random

import torch

from hackrl.core import nest


class DQN(torch.nn.Module):
    def __init__(self, qnet, target_qnet):
        super(DQN, self).__init__()
        self.qnet = qnet
        self.qnet.train()

        self.target_qnet = target_qnet
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.num_updates = 0
        self.target_update = self.qnet.flags.target_update

        self.eps_start = self.qnet.flags.eps_start
        self.eps_end = self.qnet.flags.eps_end
        self.eps_decay = self.qnet.flags.eps_decay

        self.device = self.qnet.flags.device
        self.num_actions = self.qnet.num_actions

        self.eval_mode = self.qnet.flags.eval_mode

    @property
    def version(self):
        return self.qnet.version

    @version.setter
    def version(self, v):
        self.qnet.version = v

    def initial_state(self, *args, **kwargs):
        return (
            self.qnet.initial_state(*args, **kwargs),
            self.target_qnet.initial_state(*args, **kwargs),
        )

    def forward(self, inputs, core_state):
        inputs = nest.map(lambda x: x.to(self.device), inputs)
        qnet_core, target_qnet_core = core_state
        if self.training:
            self.num_updates += 1

            qnet_output, qnet_core = self.qnet(inputs, qnet_core)
            with torch.no_grad():
                target_qnet_output, target_qnet_core = self.target_qnet(
                    inputs, target_qnet_core
                )

            qnet_output["target_qvalue"] = target_qnet_output["qvalue"]

        else:
            qnet_output, qnet_core = self.qnet(inputs, qnet_core)
            with torch.no_grad():
                target_qnet_output, target_qnet_core = self.target_qnet(
                    inputs, target_qnet_core
                )

            epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.num_updates / self.eps_decay
            )

            if not self.eval_mode and random.random() < epsilon:
                qnet_output["action"] = torch.randint(
                    0, self.num_actions, qnet_output["action"].shape
                )

        if self.num_updates % self.target_update == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        return qnet_output, (qnet_core, target_qnet_core)
