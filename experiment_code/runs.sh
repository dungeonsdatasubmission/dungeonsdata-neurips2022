#!/bin/bash

# The following are the commands used to run the experiment code on SLURM
# In particular note that 
#   1) --broker arguments are the IP & Port for a moolib broker
#   2) --constraint and --cpu arguments reflect the machine constraints used
#   3) --time is set for 3 days, although in practice many of these finish well before.
#   4) --exp_set and --exp_point are just arguments to label the experimental runs

# APPO Experiments
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal'

# Behavioural Cloning Experiments
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-AA-BC     num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True 
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=monk-AA-BC   num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True character='mon-hum-neu-mal'
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-NAO-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True  dataset_bootstrap_path=/path/to/checkpoint.tar

# APPO + Behavioural Cloning Experiments
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=@-APPO-AA-BC
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=monk-APPO-AA-BC character='mon-hum-neu-mal'  
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=@-APPO-NAO-crudeBC-all dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True num_actor_cpus=20 dataset_bootstrap_path=/path/to/checkpoint.tar

# APPO +  Kickstarting Experiments
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-APPO-AA-KS      total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/fierce-snail/checkpoint.tar
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=monk-APPO-AA-KS   total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/blazing-slug/checkpoint.tar  character='mon-hum-neu-mal'  
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-APPO-NAO-KS-all total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/celadon-llama/checkpoint.tar  dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True

