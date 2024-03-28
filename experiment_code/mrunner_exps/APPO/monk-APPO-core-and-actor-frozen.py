from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

# take configuration name without .py extension
name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": True
}

# params different between exps
unfreeze_step = 0
params_grid = [
    {
        "unfreeze_actor_steps": [unfreeze_step],
        "group": [f"{name}_{unfreeze_step}M_{i}"],
    } for i in range(6,10)
]
unfreeze_step = 10_000_000
params_grid += [
    {
        "unfreeze_actor_steps": [unfreeze_step],
        "group": [f"{name}_{unfreeze_step}M_{i}"],
    } for i in range(6,10)
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
    exclude_git_files=False,
)
