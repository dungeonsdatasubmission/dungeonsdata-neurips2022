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
from . import tasks
from . import wrappers


def create_env(flags, savedir=None, save_ttyrec_every=0):
    env_class = tasks.ENVS[flags.env.name]

    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        # "specials",
        # "colors",
        # "chars",
        # "glyphs",
        # "inv_glyphs",
        # "inv_strs",
        # "inv_letters",
        # "inv_oclasses",
    )
    kwargs = dict(
        savedir=None,
        character=flags.character,
        max_episode_steps=flags.env.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=flags.penalty_step,
        penalty_time=flags.penalty_time,
        penalty_mode=flags.fn_penalty_step,
        no_progress_timeout=150,
    )
    if flags.env in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=flags.reward_win, reward_lose=flags.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring flags.reward_win and flags.reward_lose")
    if flags.state_counter != "none":
        kwargs.update(state_counter=flags.state_counter)
    if savedir:
        kwargs.update(savedir=savedir, save_ttyrec_every=save_ttyrec_every)
    env = env_class(**kwargs)

    if flags.add_image_observation:
        env = wrappers.RenderCharImagesWithNumpyWrapperV2(
            env,
            crop_size=flags.crop_dim,
            rescale_font_size=(flags.pixel_size, flags.pixel_size),
        )

    return env
