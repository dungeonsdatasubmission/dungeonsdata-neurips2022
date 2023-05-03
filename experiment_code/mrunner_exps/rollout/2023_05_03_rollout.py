from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations
from random_word import RandomWords

from hackrl.rollout_dataset import parse_args


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

args = vars(parse_args())
args.update(config)

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

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    r = RandomWords().get_random_word()
    cfg["group"] = [f"{name}_{e}_{r}"]
    final_grid.append(dict(cfg))


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_rollout.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=args,
    params_grid=final_grid,
) 