import functools
import itertools
import os

import gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import hackrl
from hackrl.models import create_model

TEST_FLAGS = {
    "glyph_type": ["group_id", "color_char", "all", "all_cat", "full"],
    "model": ["BaselineNet"],
    "device": ["cpu"],
}


@functools.lru_cache(maxsize=32)
def create_default_flags(model):
    if model == "BaselineNet":
        return OmegaConf.load(
            os.path.join(os.path.dirname(hackrl.__file__), "models", "baseline.yaml")
        )
    raise ValueError(model)


def generate_flag_overrides():
    # [{'glyph_type': 'group_id', 'model': 'baseline', ...}, {...}]
    return [
        dict(items)
        for items in itertools.product(
            *[[(key, v) for v in value] for key, value in TEST_FLAGS.items()]
        )
    ]


@pytest.mark.parametrize("overrides", generate_flag_overrides())
def test_model_integration(overrides):
    """Test the model builds and runs by creating the model, running it and doing
    one gradient step
    """
    flags = create_default_flags(overrides["model"]).copy()
    flags.update(overrides)
    model = create_model(flags, flags["device"])
    env = gym.make("NetHack-v0")

    state = env.reset()
    state["done"] = np.array(0)
    batched_state = {
        k: torch.tensor(v.reshape((1, 1) + v.shape)) for k, v in state.items()
    }

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    optimizer.zero_grad()

    output, core_state = model(batched_state, model.initial_state())
    torch.sum(output["policy_logits"]).backward()  # Nonsense but gradients
