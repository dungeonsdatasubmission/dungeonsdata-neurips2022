from pathlib import Path

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
    "exp_tags": ["monk-APPO-AA-BC"], 
}
config = combine_config_with_defaults(config)

root_dir = Path("/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/03_05-13_30-eager_pike")


# params different between exps
params_grid = [
    {
        "checkpoint_dir": [str(root_dir / f"monk-appo-aa-bc_buxc_{i}/checkpoint/hackrl/nle/monk-APPO-AA-BC_{i}//")],
        # important, we need the same group as experiment we want to compare with
        'group': [f"monk-APPO-AA-BC_{i}"], 
        # important, it is best to set for the same name as experiment we want to compare with
    } for i in range(5)
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