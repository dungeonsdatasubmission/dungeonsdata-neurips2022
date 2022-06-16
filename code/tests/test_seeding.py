#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import random
import sys

import numpy as np
import pytest

from hackrl import tasks

seed_list_to_dict = None  # Make linter happy.


def get_nethack_env_ids():
    return [k for k in tasks.ENVS.keys() if k != "mock"]


def init_env(spec, *args, **kwargs):
    return tasks.ENVS[spec](*args, **kwargs)


def rollout_env(env, max_rollout_len):
    """Produces a rollout and asserts step outputs.

    Does *not* assume that the environment has already been reset.
    """
    obs = env.reset()
    assert env.observation_space.contains(obs)

    for _ in range(max_rollout_len):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if done:
            break
    env.close()


def compare_rollouts(env0, env1, max_rollout_len):
    """Checks that two active environments return the same rollout.

    Assumes that the environments have already been reset.
    """
    step = 0
    while True:
        a = random.choice(range(len(env0._actions)))
        obs0, reward0, done0, win0 = env0.step(a)
        obs1, reward1, done1, win1 = env1.step(a)
        step += 1
        np.testing.assert_equal(obs0, obs1)
        assert reward0 == reward1
        assert done0 == done1

        if done0 or step >= max_rollout_len:
            return


@pytest.mark.parametrize("env_name", get_nethack_env_ids())
class TestGymEnv:
    @pytest.fixture(autouse=True)  # will be applied to all tests in class
    def make_cwd_tmp(self, tmpdir):
        """Makes cwd point to the test's tmpdir."""
        with tmpdir.as_cwd():
            yield

    def test_init(self, env_name):
        """Tests default initialization given standard env specs."""
        init_env(env_name)

    def test_reset(self, env_name):
        """Tests default initialization given standard env specs."""
        env = init_env(env_name)
        env.reset()


@pytest.mark.parametrize("env_name", get_nethack_env_ids())
@pytest.mark.parametrize("rollout_len", [100])
class TestGymEnvRollout:
    @pytest.fixture(autouse=True)  # will be applied to all tests in class
    def make_cwd_tmp(self, tmpdir):
        """Makes cwd point to the test's tmpdir."""
        with tmpdir.as_cwd():
            yield

    def test_rollout(self, env_name, rollout_len):
        """Tests rollout_len steps (or until termination) of random policy."""
        env = init_env(env_name)
        rollout_env(env, rollout_len)

    def test_seed_interface_output(self, env_name, rollout_len):
        """Tests whether env.seed output can be reused correctly."""
        pytest.skip("Seeding isn't currently supported")

        env0 = init_env(env_name)
        env1 = init_env(env_name)

        seed_list0 = env0.seed()
        env0.reset()

        seed_dict = seed_list_to_dict(seed_list0)
        assert env0.get_seeds() == seed_dict

        seed_list1 = env1.seed(seed_dict)
        assert seed_list0 == seed_list1

    def test_seed_rollout_seeded(self, env_name, rollout_len):
        """Tests that two seeded envs return same step data."""
        pytest.skip("Seeding isn't currently supported")

        env0 = init_env(env_name)
        env1 = init_env(env_name)

        env0.seed()
        obs0 = env0.reset()
        seeds0 = env0.get_seeds()

        env1.seed(seeds0)
        obs1 = env1.reset()
        seeds1 = env1.get_seeds()

        assert seeds0 == seeds1

        np.testing.assert_equal(obs0, obs1)
        compare_rollouts(env0, env1, rollout_len)

    def test_seed_rollout_seeded_int(self, env_name, rollout_len):
        """Tests that two seeded envs return same step data."""
        pytest.skip("Seeding isn't currently supported")

        env0 = init_env(env_name)
        env1 = init_env(env_name)

        env0.seed(random.randrange(sys.maxsize))
        obs0 = env0.reset()
        seeds0 = env0.get_seeds()

        env1.seed(seeds0)
        obs1 = env1.reset()
        seeds1 = env1.get_seeds()

        assert seeds0 == seeds1

        np.testing.assert_equal(obs0, obs1)
        compare_rollouts(env0, env1, rollout_len)
