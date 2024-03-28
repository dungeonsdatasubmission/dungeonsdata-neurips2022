#!/bin/bash

export BROKER_IP=0.0.0.0
export BROKER_PORT=4431

python -m moolib.broker &

# python -m debugpy --wait-for-client --listen 5678 ./hackrl/experiment.py connect=$BROKER_IP:$BROKER_PORT exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal'

# Decision Transformer Experiments
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT model=DecisionTransformer use_prev_action=false exp_set=2G exp_point=@-AA-DT num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True group='@-AA-DT'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT model=DecisionTransformer use_prev_action=false exp_set=2G exp_point=monk-AA-DT num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True character='mon-hum-neu-mal' group='monk-AA-DT'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT model=DecisionTransformer use_prev_action=false exp_set=2G  exp_point=@-NAO-DT num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True  dataset_bootstrap_path=/path/to/checkpoint.tar group='@-NAO-DT'

# APPO Experiments
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 group='@-APPO'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal' group='monk-APPO'

# Behavioural Cloning Experiments
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G exp_point=@-AA-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True group='@-AA-BC'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G exp_point=monk-AA-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True character='mon-hum-neu-mal' group='monk-AA-BC'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G exp_point=@-NAO-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True  dataset_bootstrap_path=/path/to/checkpoint.tar group='@-NAO-BC'

# APPO + Behavioural Cloning Experiments
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_point=monk-APPO-AA-BC num_actor_cpus=20 total_steps=2_000_000_000 batch_size=128 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G character='mon-hum-neu-mal' group='monk-APPO-AA-BC'

# APPO +  Kickstarting Experiments
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G exp_point=monk-APPO-AA-KS total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/net/pr2/projects/plgrid/plgggmum_crl/bcupial/monk-AA-BC/checkpoint_v60242.tar  character='mon-hum-neu-mal' group='monk-APPO-AA-KS'
# python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G exp_point=monk-APPO-AA-KS total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/home/z1188643/dungeonsdata-neurips2022/experiment_code/checkpoint/hackrl/nle/monk-APPO-AA-BC-completed/checkpoint_v482821.tar  character='mon-hum-neu-mal' group='monk-APPO-AA-KS'