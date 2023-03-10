#!/bin/bash

#SBATCH --job-name=NetHackRL
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=big  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgxa100  # dgxa100 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"

singularity exec --nv /shared/results/z1188643/dungeons/dungeons.sif ./train.sh
