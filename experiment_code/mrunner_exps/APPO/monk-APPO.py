from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    'group': "monk-APPO",
    "character": "mon-hum-neu-mal",
}


# params different between exps
params_grid = [
    {
        "seed": [0, 1, 2],
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    cfg["group"] = [f"{name}_{e}"]
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
