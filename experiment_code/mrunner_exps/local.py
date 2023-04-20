from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_point": "monk-AA-BC",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 16,
    "batch_size": 32,
    "ttyrec_batch_size": 64,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    "character": "mon-hum-neu-mal",
    'group': "monk-AA-BC",
    "connect":"0.0.0.0:4431",
}


# params different between exps
params_grid = []

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
)
