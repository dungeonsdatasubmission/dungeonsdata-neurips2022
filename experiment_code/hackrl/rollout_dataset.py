import os
import argparse

from pathlib import Path

import moolib
import numpy as np
import omegaconf
import torch
import tqdm
import wandb

import hackrl.core
import hackrl.environment
import hackrl.models
from hackrl.core import nest

ENVS = None


def load_model_and_flags(path, device):
    load_data = torch.load(path, map_location=torch.device(device))
    flags = omegaconf.OmegaConf.create(load_data["flags"])
    flags.device = device
    model = hackrl.models.create_model(flags, device)
    if flags.use_kickstarting:
        print("Kickstarting")
        t_data = torch.load(flags.kickstarting_path)
        t_flags = omegaconf.OmegaConf.create(t_data["flags"])
        teacher = hackrl.models.create_model(t_flags, flags.device)
        # teacher.load_state_dict(load_data["learner_state"]["model"])
        model = hackrl.models.KickStarter(
            model, teacher, run_teacher_hs=flags.run_teacher_hs
        )
    model.load_state_dict(load_data["learner_state"]["model"])
    return model, flags


def generate_envpool_rollouts(
    model, 
    flags, 
    rollouts=512, 
    batch_size=512,
    num_actor_cpus = 20,
    num_actor_batches = 1,
    pbar_idx=0, 
    score_target=10000,
    savedir=None,
    save_ttyrec_every=1000000, # only first time
):
    global ENVS
    
    assert batch_size == rollouts
    assert num_actor_batches == 1

    # NB: We do NOT want to generate the first N rollouts from B batch
    # of envs since this will bias short episodes.
    # Instead lets just allocate some episodes to each env
    split = rollouts // (batch_size * num_actor_batches)
    flags.batch_size = batch_size
    device = flags.device

    ENVS = moolib.EnvPool(
        lambda: hackrl.environment.create_env(flags, savedir=savedir, save_ttyrec_every=save_ttyrec_every),
        num_processes=num_actor_cpus,
        batch_size=batch_size,
        num_batches=num_actor_batches,
    )

    rollouts_left = (
        torch.ones(
            (
                num_actor_batches,
                batch_size,
            )
        )
        .long()
        .to(device)
        * split
    )
    current_reward = torch.zeros(
        (
            num_actor_batches,
            batch_size,
        )
    ).to(device)
    timesteps = torch.zeros(
        (
            num_actor_batches,
            batch_size,
        )
    ).to(device)

    returns = []
    results = [None, None]
    grand_pbar = tqdm.tqdm(position=0, leave=True)
    pbar = tqdm.tqdm(
        total=batch_size * num_actor_batches * split, position=pbar_idx + 1, leave=True
    )

    action = torch.zeros((num_actor_batches, batch_size)).long().to(device)
    hs = [model.initial_state(batch_size) for _ in range(num_actor_batches)]
    hs = nest.map(lambda x: x.to(device), hs)

    totals = torch.sum(rollouts_left).item()
    subtotals = [torch.sum(rollouts_left[i]).item() for i in range(num_actor_batches)]
    while totals > 0:
        grand_pbar.update(1)
        for i in range(num_actor_batches):
            if subtotals[i] == 0:
                continue
            if results[i] is None:
                results[i] = ENVS.step(i, action[i])
            outputs = results[i].result()

            env_outputs = nest.map(lambda t: t.to(flags.device, copy=True), outputs)
            env_outputs["prev_action"] = action[i]
            current_reward += env_outputs["reward"]

            env_outputs["timesteps"] = timesteps[i]
            env_outputs["max_scores"] = (torch.ones_like(env_outputs["timesteps"]) * score_target).float()
            env_outputs["mask"] = torch.ones_like(env_outputs["timesteps"]).to(torch.bool)
            env_outputs["scores"] = current_reward[i]

            done_and_valid = env_outputs["done"].int() * rollouts_left[i].bool().int()
            finished = torch.sum(done_and_valid).item()
            totals -= finished
            subtotals[i] -= finished

            for j in np.argwhere(done_and_valid.cpu().numpy()):
                returns.append(current_reward[i][j[0]].item())

            current_reward[i] *= 1 - env_outputs["done"].int()
            timesteps[i] += 1
            timesteps[i] *= 1 - env_outputs["done"].int()
            rollouts_left[i] -= done_and_valid
            if finished:
                pbar.update(finished)

            env_outputs = nest.map(lambda x: x.unsqueeze(0), env_outputs)
            with torch.no_grad():
                outputs, hs[i] = model(env_outputs, hs[i])
            action[i] = outputs["action"].reshape(-1)
            results[i] = ENVS.step(i, action[i])
    return len(returns), np.mean(returns), np.std(returns), np.median(returns)


def evaluate_folder(name, path, device, pbar_idx, savedir, rollouts, batch_size, **kwargs):
    print(f"{pbar_idx} {name} Using: {path}")
    save_dir = Path(savedir) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    model, flags = load_model_and_flags(path, device)

    assert rollouts % batch_size == 0
    iters = rollouts // batch_size

    final_returns = []
    for i in range(iters):
        returns = generate_envpool_rollouts(
            model=model, 
            flags=flags, 
            pbar_idx=pbar_idx, 
            rollouts=batch_size,
            batch_size=batch_size,
            savedir=savedir,
            **kwargs,
        )
        final_returns.append(returns)
    
    return (name, path) + tuple(list(np.stack(final_returns).mean(axis=0)))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--savedir", type=Path)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_actor_cpus", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_target", type=float, default=5000)
    # wandb stuff
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--group", type=str, default="group2")
    parser.add_argument("--exp_tags", type=str, default="eval2")
    return parser.parse_known_args(args=args)[0]


def main(variant):
    name = variant["name"]
    checkpoint_dir = variant["checkpoint_dir"]
    savedir = variant["savedir"]
    rollouts = variant["rollouts"]
    device = variant["device"]
    log_to_wandb = variant["wandb"]
    kwargs = dict(
        batch_size=variant["batch_size"],
        num_actor_cpus=variant["num_actor_cpus"],
        score_target=variant["score_target"],
    )

    if log_to_wandb:
        wandb.init(
            project="nle",
            config=variant,
            group=variant["group"],
            entity="gmum",
            name=name,
        )    

    results = (name, checkpoint_dir, -1, -1, -1, -1)

    results = evaluate_folder( 
        name=name, 
        path=checkpoint_dir, 
        device=device, 
        pbar_idx=0, 
        savedir=savedir,
        rollouts=rollouts,
        **kwargs
    )

    stats_values = dict(
        len=results[2],
        mean=results[3],
        std=results[4],
        median=results[5],
    )

    print(
        f"{results[0]} Done {results[1]}  Mean {results[3]} ± {results[4]}  | Median {results[5]}"
    )
    if results[2] > -2:
        data = (
            rollouts,
        ) + results
        os.makedirs(f"{savedir}/{name}/", exist_ok=True)
        with open(f"{savedir}/{name}/{checkpoint_dir.split('/')[-1]}.txt", "w") as f:
            f.write(",".join(str(d) for d in data) + "\n")
    print("done")

    if wandb:
        wandb.log(stats_values)


if __name__ == "__main__":
    args = vars(parse_args())
    main(variant=vars(args))
