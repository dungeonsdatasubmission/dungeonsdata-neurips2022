from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-AA-DT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 256,
    "batch_size": 128,
    "ttyrec_batch_size": 512,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    'group': "monk-AA-DT",
    "character": "mon-hum-neu-mal",
    "model": "DecisionTransformer",
    "return_to_go": False, # TODO: test
    "use_timesteps": True,
    "use_returns": True,
    "use_timesteps": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "n_layer": 3, # TODO: test
    "n_head": 1,
    "grad_norm_clipping": 4,
    "hidden_dim": 512,
}


# params different between exps
params_grid = [
    {
        "n_layer": [6],
        "n_head": [8],
        "weight_decay": [0.1],
        "warmup_steps": [4000],
        "adam_learning_rate": [0.0003],
    },
]

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
