#!/bin/bash

export BROKER_IP=$(echo $SSH_CONNECTION | cut -d' ' -f3)
export BROKER_PORT=4431

python -m moolib.broker &
python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT
