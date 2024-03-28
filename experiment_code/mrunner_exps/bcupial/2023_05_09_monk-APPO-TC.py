from pathlib import Path
from random_word import RandomWords

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    'group': "monk-APPO",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
}

# params different between exps
params_grid = [
    {
        "seed":  list(range(5)),
        "unfreeze_actor_steps": [0],
        "use_checkpoint_actor": [True],
        "model_checkpoint_path": ["/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-11_07-elastic_mccarthy/monk-appo-t_85h5_0/checkpoint/hackrl/nle/monk-APPO-T_0_lips/checkpoint_v50000000"],
        "kickstarting_loss": [0],
        "use_kickstarting": [True],
        "log_kickstarting": [True],
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
