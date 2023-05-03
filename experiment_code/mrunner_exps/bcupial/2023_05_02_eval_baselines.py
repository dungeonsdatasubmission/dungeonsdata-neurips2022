from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations
from random_word import RandomWords

from hackrl.eval import parse_args


name = globals()["script"][:-3]


# params for all exps
config = {
    "exp_tags": [name],
    "name": "eval_monk-AA-DT-40k-newembeds",
    "checkpoint_dir": "/checkpoint/hackrl/nle/monk-AA-DT-40k-newembeds/checkpoint.tar",
    "output_dir": "eval_results",
    "num_actor_cpus": 20,
    "num_actor_batches": 2,
    "rollouts": 1024,
    "batch_size": 256,
    "wandb": True,
}

args = vars(parse_args())
args.update(config)

monk_appo = "/mscratch/nle/02_05-18_04-musing_bartik/monk-appo_wdkc_2/checkpoint/hackrl/nle/monk-APPO_2/checkpoint.tar"
monk_appo_t = "/mscratch/nle/02_05-11_47-blissful_noyce/monk-appo-t_j13x_2/checkpoint/hackrl/nle/monk-APPO-T_2/checkpoint.tar"
monk_appo_aa_bc = "/mscratch/nle/02_05-18_05-jovial_banach/monk-appo-aa-bc_rz35_0/checkpoint/hackrl/nle/monk-APPO-AA-BC_0/checkpoint.tar"
monk_appo_aa_bc_t = "/mscratch/nle/01_05-19_47-sad_bohr/monk-appo-aa-bc-t_v0r1_4/checkpoint/hackrl/nle/monk-APPO-AA-BC-T_4/checkpoint.tar"
monk_appo_aa_ks = "/mscratch/nle/02_05-18_05-eloquent_ramanujan/monk-appo-aa-ks_ic86_1/checkpoint/hackrl/nle/monk-APPO-AA-KS_1/checkpoint.tar"
monk_appo_aa_ks_t = "/mscratch/nle/01_05-19_48-vigilant_dijkstra/monk-appo-aa-ks-t_0k6i_4/checkpoint/hackrl/nle/monk-APPO-AA-KS-T_4/checkpoint.tar"

# params different between exps
params_grid = [
    {
        "rollouts": [1024],
        "batch_size": [256],
        "checkpoint_dir": [
            monk_appo,
            monk_appo_t,
            monk_appo_aa_bc,
            monk_appo_aa_bc_t,
            monk_appo_aa_ks,
            monk_appo_aa_ks_t,
        ]
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
    script="python3 mrunner_eval.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=args,
    params_grid=final_grid,
) 