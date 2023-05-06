from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPODT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    'group': "monk-APPODT",
    "character": "mon-hum-neu-mal",
    "model": "DecisionTransformer",
    "use_timesteps": True,
    "use_timesteps": True,
    "return_to_go": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "grad_norm_clipping": 4,
    "n_layer": 6,
    "n_head": 8,
    "hidden_dim": 512,
    "warmup_steps": 10000,
    "weight_decay": 0.01,
}


# params different between exps
params_grid = {
    "actor_batch_size": [128],
    "batch_size": [64],
    "virtual_batch_size": [64],
    "ttyrec_batch_size": [256],
    "dbfilename": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
    "dataset": ["bc"],
    "wandb": [True],

    "use_checkpoint_actor": [False],
    "unfreeze_actor_steps": [0],
    "model_checkpoint_path": ["/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-DT/checkpoint.tar"],
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
)
