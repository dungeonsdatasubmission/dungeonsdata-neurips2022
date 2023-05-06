from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_exps.evaluation.utils import combine_config_with_defaults
# if error "there is no mrunner_exps" module is raised just run 
# export PYTHONPATH=. 
# or
# export PYTHONPATH=experiment_code

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": name,
    "run_kind": "eval",
    "name": "eval",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "wandb": False,
    'group': "monk-APPODT",
    "checkpoint_dir": "/path/to/checkpoint/file",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "rollouts": [16],
        "batch_size": [4],
        "checkpoint_dir": ["/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-BC/checkpoint.tar"],
        "wandb": [True],
    },
]


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_eval.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
) 