from pathlib import Path
import numpy as np
from mrunner.helpers.specification_helper import create_experiments_helper
from hackrl.eval import parse_args

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "name": "eval_monk-AA-DT-40k-newembeds",
    "checkpoint_dir": "/checkpoint/hackrl/nle/monk-AA-DT-40k-newembeds/checkpoint.tar",
    "output_dir": "DT_results",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "wandb": True,
}

args = vars(parse_args())
args.update(config)

# params different between exps
params_grid = {
    "score_target": list(np.logspace(2.0, 5.0, num=10)),
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_eval.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=args,
    params_grid=params_grid,
)
