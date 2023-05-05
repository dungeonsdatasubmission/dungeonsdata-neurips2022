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
from typing import List

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
    def _initialize_weights(layer, gain=1.0):
        if hasattr(layer, "bias") and isinstance(
            layer.bias, torch.nn.parameter.Parameter
        ):
            layer.bias.data.fill_(0)

        init_true = isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) or (
            isinstance(layer, torch.jit.RecursiveScriptModule)
            and (layer.original_name == "Conv2d" or layer.original_name == "Linear")
        )

        if flags.initialisation == "orthogonal":
            if init_true:
                torch.nn.init.orthogonal_(layer.weight.data, gain=gain)
        elif flags.initialisation == "xavier_uniform":
            if init_true:
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=gain)
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

    if flags['use_checkpoint_actor']:

        def distil_actor_nad_core(load_data):
            return {
                key: elem
                for key, elem in load_data["learner_state"]["model"].items()
                if "baseline" not in key
            }

        load_data = torch.load(
            flags["model_checkpoint_path"],
            map_location=torch.device(device),
        )
        model.load_state_dict(distil_actor_nad_core(load_data), strict=False)
        freeze(model)
        unfreeze_selected(model, ["baseline", "embed_ln"])

    return model


def load_model(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_model(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model


def set_requires_grad(model, modules: List[str], requires_grad: bool):
    for module_name in modules:
        for name, param in model.named_parameters():
            if module_name in name:
                param.requires_grad = requires_grad


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def freeze_selected(model, modules: List[str]):
    set_requires_grad(model, modules, False)


def unfreeze_selected(model, modules: List[str]):
    set_requires_grad(model, modules, True)
