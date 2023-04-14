# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from nle.env.base import DUNGEON_SHAPE
from omegaconf import OmegaConf

from ..tasks import ENVS
from .baseline import BaselineNet
from .chaotic_dwarf import ChaoticDwarvenGPT5
from .decision_transformer import DecisionTransformer
from .offline_chaotic_dwarf import DQNChaoticDwarvenGPT5, IQLChaoticDwarvenGPT5 
from .inverse_model import BigInverseOnlyModel
from .kickstarter import KickStarter
from .dqn import DQN
from .cql import CQL
from .iql import IQL

MODELS = [
    BaselineNet,
    ChaoticDwarvenGPT5,
    KickStarter,
    BigInverseOnlyModel,    
    DQNChaoticDwarvenGPT5, 
    IQLChaoticDwarvenGPT5, 
    DQN, 
    CQL, 
    IQL,
    DecisionTransformer,    
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}


def initialize_weights(flags, model):
    def _initialize_weights(layer):
        if hasattr(layer, "bias") and isinstance(
            layer.bias, torch.nn.parameter.Parameter
        ):
            layer.bias.data.fill_(0)

        if flags.initialisation == "orthogonal":
            if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.orthogonal_(layer.weight.data, gain=1.0)
        elif flags.initialisation == "xavier_uniform":
            if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=1.0)
            else:
                pass
        else:
            pass

    model.apply(_initialize_weights)


def create_model(flags, device):
    try:
        model_cls = MODELS_LOOKUP[flags.model]
    except KeyError:
        raise NotImplementedError("model=%s" % flags.model) from None

    action_space = ENVS[flags.env.name](savedir=None).actions

    model = model_cls(DUNGEON_SHAPE, action_space, flags, device)
    model.to(device=device)

    initialize_weights(flags, model)
    return model


def load_model(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_model(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model
