import argparse
import os

from pathlib import Path

import numpy as np
import wandb

from hackrl.eval import evaluate_folder

os.environ["MOOLIB_ALLOW_FORK"] = "1"


def log(results, step):
    returns = results["returns"]
    steps = results["steps"]
    scores = results["scores"]
    times = results["times"]

    wandb.log(        
        {
            "eval/mean_episode_return": np.mean(returns),
            "eval/std_episode_return": np.std(returns),
            "eval/median_episode_return": np.median(returns),
            "eval/mean_episode_steps": np.mean(steps),
            "eval/std_episode_steps": np.std(steps),
            "eval/median_episode_steps": np.median(steps),

            "eval/mean_episode_scores": np.mean(scores),
            "eval/std_episode_scores": np.std(scores),
            "eval/median_episode_scores": np.median(scores),
            "eval/mean_episode_times": np.mean(times),
            "eval/std_episode_times": np.std(times),
            "eval/median_episode_times": np.median(times),
        },
        step=step
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="evaluation_array")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--checkpoint_step", type=int, default=100_000_000)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--num_actor_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--group", type=str)
    parser.add_argument("--exp_tags", type=str)
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

    if log_to_wandb:
        wandb.init(
            project="nle",
            config=variant,
            group=variant["group"],
            entity="gmum",
            name=name,
        )

    checkpoints = list(Path(checkpoint_dir).iterdir())
    checkpoints = list(filter(lambda path: path.name.startswith("checkpoint_"), checkpoints))
    checkpoints = list(filter(lambda path: int(path.name.split('_')[1][1:]) % variant["checkpoint_step"] == 0, checkpoints))
    checkpoints = sorted(checkpoints, key=lambda path: int(path.name.split('_')[1][1:]))

    for e, checkpoint in enumerate(checkpoints):
        print(f"Evaluating checkpoint {checkpoint}")

        step = int(checkpoint.name.split('_')[1][1:])

        results = evaluate_folder(
            pbar_idx=e, 
            path=checkpoint, 
            **kwargs,
        )

        if log_to_wandb:
            log(results, step)


if __name__ == "__main__":
    args = vars(parse_args())
    main(variant=vars(args))
