from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.eval_array import parse_args as eval_array_parse_args
from hackrl.rollout import parse_args as rollout_parse_args


PARSE_ARGS_DICT = {"eval": eval_parse_args, "eval_array": eval_array_parse_args, "rollout": rollout_parse_args}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))
    res.update(config)
    return res

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_kind": "eval_array",
    "name": "eval_array",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "checkpoint_step": 100_000_000,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/dir",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "rollouts": [16],
        "batch_size": [4],
        "checkpoint_dir": ["/home/bartek/Workspace/CW/dungeonsdata-neurips2022/checkpoint/hackrl/nle/monk-APPODT"],
        'group': ["monk-APPODT"], # <- important, we need the same group as experiment we want to compare with
        "exp_tags": ["local"], # <- important, it is best to set for the same name as experiment we want to compare with
        "checkpoint_step": [100_000_000],
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