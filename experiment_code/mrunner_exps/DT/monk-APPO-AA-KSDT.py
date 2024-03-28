from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AA-KSDT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "ttyrec_batch_size": 256,
    "kickstarting_loss": 0.1,
    'group': name,
    "use_kickstarting": True, 
    "kickstarting_path": "/scratch/nle/25_04-10_53-romantic_davinci/2023-04-25-search-layer-head_wxxn_0/checkpoint/hackrl/nle/2023_04_25_search_layer_head_0/checkpoint.tar",
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
