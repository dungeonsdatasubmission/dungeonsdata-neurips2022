from random_word import RandomWords

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AA-KL",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-AA-KL",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
    "ttyrec_batch_size": 256,
    "kickstarting_loss_bc": 0.1,
    "use_kickstarting_bc": True, 
    "kickstarting_path": "/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/monk-AA-BC/checkpoint.tar",
    "dataset": "bc1",
}


# params different between exps
params_grid = [
    {
        "seed":  list(range(5)),
        "kickstarting_loss_bc": [0.5],
        # load from checkpoint
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "model_checkpoint_path": ["/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/monk-AA-BC/checkpoint.tar"],
        # log forgetting
        "log_forgetting": [True],
        "forgetting_dataset": ["bc1"],
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
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=final_grid,
)
