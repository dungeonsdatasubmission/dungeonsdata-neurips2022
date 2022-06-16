import math
import random

import torch

from hackrl.core import nest


class IQL(torch.nn.Module):
    def __init__(self, iql_net, target_iql_net):
        super(IQL, self).__init__()
        self.iql_net = iql_net
        self.iql_net.train()

        self.target_iql_net = target_iql_net
        self.target_iql_net.load_state_dict(self.iql_net.state_dict())
        self.target_iql_net.eval()

        self.tau = self.iql_net.flags.tau
        self.device = self.iql_net.flags.device

    @property
    def version(self):
        return self.iql_net.version

    @version.setter
    def version(self, v):
        self.iql_net.version = v

    def initial_state(self, *args, **kwargs):
        return (
            self.iql_net.initial_state(*args, **kwargs),
            self.target_iql_net.initial_state(*args, **kwargs),
        )
        
    def update_target_critic(self):
        for target_param, local_param in zip(self.target_iql_net.parameters(), self.iql_net.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def forward(self, inputs, core_state):
        inputs = nest.map(lambda x: x.to(self.device), inputs)

        iql_core, target_iql_core = core_state

        iql_output, iql_net_core = self.iql_net(inputs, iql_core)

        with torch.no_grad():
            target_iql_output, target_iql_core = self.target_iql_net(
                inputs, target_iql_core
            )

        iql_output["target_q1"] = target_iql_output["q1"]
        iql_output["target_q2"] = target_iql_output["q2"]

        return iql_output, (iql_core, target_iql_core)
        