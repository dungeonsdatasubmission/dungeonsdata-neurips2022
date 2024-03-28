from pathlib import Path
from random_words import RandomWords
from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AA-KS-BC",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "ttyrec_batch_size": 256,
    "supervised_loss": 0,
    "behavioural_clone": False,
    "use_kickstarting": False, 
    "kickstarting_loss": 0.1,
    "kickstarting_decay": 1,
    'group': "monk-APPO-AA-BC",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
    "use_kickstarting_bc": True,
    "kickstarting_path": "/net/tscratch/people/plgmostaszewski/dungeonsdata-neurips2022/experiment_code/monk-AA-BC/checkpoint.tar",
}


# params different between exps
params_grid = [
    {
       
        "seed":  list(range(5)),
        "dataset": ["bc1"],
        "kickstarting_loss": [0.1,0.5],
        "kickstarting_decay": [1],
        "unfreeze_actor_steps": [0,50_000_000],
        "use_checkpoint_actor": [True],
        "model_checkpoint_path": ["/net/tscratch/people/plgmostaszewski/dungeonsdata-neurips2022/experiment_code/monk-AA-BC/checkpoint.tar"],
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    r = RandomWords().random_word()
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
