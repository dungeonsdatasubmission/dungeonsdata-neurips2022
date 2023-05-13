from mrunner.helpers.specification_helper import create_experiments_helper

from hackrl.eval import parse_args as eval_parse_args
from hackrl.rollout import parse_args as rollout_parse_args


PARSE_ARGS_DICT = {
    "eval": eval_parse_args,
    "rollout": rollout_parse_args,
}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))
    res.update(config)
    return res


name = globals()["script"][:-3]


# params for all exps
config = {
    "exp_tags": [name],
    "run_kind": "rollout",
    "name": "rollout",
    "num_actor_cpus": 1,
    "num_actor_batches": 1,
    "rollouts": 1024,
    "batch_size": 1,
    "wandb": True,
    "checkpoint_dir": "/path/to/checkpoint/file",
}
config = combine_config_with_defaults(config)

# params different between exps
params_grid = [
    {
        "checkpoint_dir": ["/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/10_05-09_22-awesome_heisenberg/monk-aa-bc-deep_hp0i_0/checkpoint/hackrl/nle/monk-AA-BC_deep_0/checkpoint.tar"],
        "savedir": ["/nle/nld-bc-deep/nld_data"],
    },
    {
        "checkpoint_dir": ["/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/10_05-09_22-awesome_heisenberg/monk-aa-bc-deep_hp0i_5/checkpoint/hackrl/nle/monk-AA-BC_deep_5/checkpoint.tar"],
        "savedir": ["/nle/nld-bc-midscore/nld_data"],
    },
    {
        "checkpoint_dir": ["/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/10_05-09_22-awesome_heisenberg/monk-aa-bc-deep_hp0i_10/checkpoint/hackrl/nle/monk-AA-BC_deep_10/checkpoint.tar"],
        "savedir": ["/nle/nld-bc-highscore/nld_data"],
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
