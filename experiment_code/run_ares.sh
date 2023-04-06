#!/usr/bin/env bash

#SBATCH -o logger-%j.txt
#SBATCH --gres gpu:1  
#SBATCH --nodes 1
#SBATCH --mem=20G
#SBATCH -c 20
#SBATCH -t 2880
#SBATCH -A plgmodernclgpu-gpu
#SBATCH -p plgrid-gpu-v100


set -e


module load cuda/11.6.0
export PYTHONPATH=$PYTHONPATH:.

singularity exec --nv -H $PWD:/homeplaceholder --env WANDB_API_KEY=d2f9309c1cee36dc7ad726c57e4eba04974d9914 --env WANDBPWD=$PWD -B /net/ascratch/people/plgbartekcupial/nle:/nle -B $TMPDIR:/tmp /net/ascratch/people/plgbartekcupial/nle/dungeons.sif ./train.sh