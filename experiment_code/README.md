# Code for Dungeons & Data

This code will be tidied up and released as an easily reproduced baseline on NLE, when the dataset is released.


## Setup


```sh
# Basic setup, ensure you already have Cuda 10.2+ and CUDNN installed.
conda create -n nle python=3.9
conda activate nle

# Install PyTorch and cmake.
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install cmake

# Install moolib
pip install git+ssh://git@github.com/facebookresearch/moolib

# Install NLE .
pip install git+https://github.com/facebookresearch/nle.git@main

# Get this repo.
git clone --recursive git+https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022.git 
cd dungeonsdata-neurips2022/experiment_code

# install render_utils, hackrl 
pip install -r requirements.txt
cd render_utils && pip install -e . && cd ..
pip install -e .

# Test NLE.
python -c 'import gym; import nle; env = gym.make("NetHackScore-v0"); env.reset(); env.render()'



```

## Running the broker

To run an experiment with many peers on different machines, these peers need
to be able to find each other. We use the moolib _broker_ for that purpose.

First, start a moolib broker in a shell on your devfair:

```
python -m moolib.broker
```

It will output something like `Broker listening at 0.0.0.0:4431`.



As an example:

```
export BROKER_IP=$(echo $SSH_CONNECTION | cut -d' ' -f3)  # Should give your machines's IP.
export BROKER_PORT=4431
```

Note that a **single broker is enough** for all your experiments, as long as
the combination of `project` and `group` flags of each experiment are unique.

## Run a local experiments

With this information and a running broker, we can start a local experiment:

```
# Run an experiment locally using default arguments.
python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT
```

By setting `wandb: true` in `hackrl/config.yaml`,
you can check learning curves on Weights and Biases.

## Run a larger-scale experiment on a SLURM cluster

Keep your broker running and do

```
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --dry
```

Remove the `--dry` to actually run this. You can check the status of your experiments with

```
squeue -u$USER
```

You can use `scancel` to stop experiments, but be careful with that command.

