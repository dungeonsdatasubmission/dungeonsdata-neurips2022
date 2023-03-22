#!/bin/bash

export BROKER_IP=0.0.0.0
export BROKER_PORT=4431

python -m moolib.broker &
python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_point=@-AA-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True 
