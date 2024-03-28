import argparse
import json
import logging
import shutil
import tempfile

from pathlib import Path

import omegaconf
import wandb

from hackrl.eval import evaluate_folder, results_to_dict


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="rollout")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--results_path", type=Path, default="data.json")
    parser.add_argument("--savedir", type=Path)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    parser.add_argument("--save_ttyrec_every", type=int, default=1)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--exp_kind", type=str, default="eval")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    log_to_wandb = variant["wandb"]

    kwargs = dict(
        device=variant["device"],
        rollouts=variant["rollouts"],
        batch_size=variant["batch_size"],
        num_actor_cpus=variant["num_actor_cpus"],
        num_actor_batches=variant["num_actor_batches"],
        score_target=variant["score_target"],
        log_to_wandb=log_to_wandb,
    )

    print(f"Rollouting checkpoint {checkpoint_dir}")

    results, flags, step = evaluate_folder(
        pbar_idx=0,
        path=checkpoint_dir,
        savedir=variant["savedir"],
        save_ttyrec_every=variant["save_ttyrec_every"],
        **kwargs,
    )

    config = omegaconf.OmegaConf.to_container(flags)
    config.update(variant)

    if log_to_wandb:
        wandb.init(
            project="nle",
            config=config,
            group=config["group"],
            entity="gmum",
            name=name,
        )
        wandb.log(results, step=step)

    with open(variant["results_path"], "w") as file:
        json.dump(results_to_dict(results), file)


if __name__ == "__main__":
    tempdir = tempfile.mkdtemp()
    tempfile.tempdir = tempdir

    try:
        args = vars(parse_args())
        main(variant=args)
    finally:
        logging.info(f"Removing all temporary files in {tempfile.tempdir}")
        shutil.rmtree(tempfile.tempdir)
