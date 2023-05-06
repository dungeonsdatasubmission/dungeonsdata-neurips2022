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
    "exp_tags": [name],
    "name": "rollout",
    "checkpoint_dir": "/checkpoint/hackrl/nle/monk-AA-DT-40k-newembeds/checkpoint.tar",
    "savedir": "rollout_results",
    "num_actor_cpus": 20,
    "rollouts": 8192,
    "batch_size": 256,
    "wandb": True,
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": ["/checkpoint/hackrl/nle/monk-AA-BC_1/checkpoint.tar"],
        "savedir": ["/nle/nld-bc/nld_data"],
    },
    {
        "checkpoint_dir": ["/tscratch/nle/30_04-06_07-relaxed_spence/2023-04-30-pretrain-no-returns_usal_1/checkpoint/hackrl/nle/2023_04_30_pretrain_no_returns_1/checkpoint.tar"],
        "savedir": ["/nle/nld-dt/nld_data"],
    },
]


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_rollout.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
) 
