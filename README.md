# Dungeon And Data

Dataset Instructions and Tutorials for Submission to NeurIPS2022 Datasets and Benchmarks Track

## License

This data is licensed under the NetHack General Public License - based on the GPL-style BISON license. It is the license used for the game of NetHack, and can be found [here](https://github.com/facebookresearch/nle/blob/main/LICENSE).


## Installation

Inspired with nle installation. Other dependencies can be installed by doing:

```bash
apt-get -y install git build-essential ffmpeg python3-pip \ python3-dev  python3-numpy curl libgl1-mesa-dev libgl1-mesa-glx autoconf libtool pkg-config libglew-dev libosmesa6-dev libbz2-dev libclang-dev software-properties-common net-tools unzip vim wget xpra xserver-xorg-dev virtualenv tmux make gcc g++
```

We advise using a conda environment or a singularity image. Singularity definition can be found in `experiment_code/assets`. Setting up with conda can be done by.

```bash
cd experiment_code

conda create -y -n dungeons python=3.9
conda activate dungeons

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cmake flex bison
conda install pybind11 -c conda-forge
conda install cudnn

pip install tqdm debugpy
pip install git+https://github.com/facebookresearch/moolib
pip install git+https://github.com/facebookresearch/nle
pip install -r requirements.txt 

pybind11_INCLUDE_DIR=$(dirname $(dirname $(which conda)))/envs/dungeons/share/cmake/pybind11

cd render_utils && pip install -e . && cd ..

pip install -e .
```

## Accessing the Dataset

The dataset is currently hosted on FAIRs public s3 bucket. You can access it it via a browser or curls below.

#### TasterPacks

We provide a small "taster pack" dataset, that contain a random subsample of the full datasets, to allow fast iteration for those looking to play around with `nld`.
- [`nld-aa-taster.zip`](https://dl.fbaipublicfiles.com/nld/nld-aa-taster/nld-aa-taster.zip
) (1.6GB)


#### Full Downloads
You can download these by visiting the links or using curl:

`NLD-AA` (16 file)
```
# Download NLD-AA
mkdir -p nld-aa
curl -o nld-aa/nld-aa-dir-aa.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aa.zip
curl -o nld-aa/nld-aa-dir-ab.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ab.zip
curl -o nld-aa/nld-aa-dir-ac.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ac.zip
curl -o nld-aa/nld-aa-dir-ad.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ad.zip
curl -o nld-aa/nld-aa-dir-ae.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ae.zip
curl -o nld-aa/nld-aa-dir-af.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-af.zip
curl -o nld-aa/nld-aa-dir-ag.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ag.zip
curl -o nld-aa/nld-aa-dir-ah.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ah.zip
curl -o nld-aa/nld-aa-dir-ai.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ai.zip
curl -o nld-aa/nld-aa-dir-aj.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aj.zip
curl -o nld-aa/nld-aa-dir-ak.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ak.zip
curl -o nld-aa/nld-aa-dir-al.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-al.zip
curl -o nld-aa/nld-aa-dir-am.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-am.zip
curl -o nld-aa/nld-aa-dir-an.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-an.zip
curl -o nld-aa/nld-aa-dir-ao.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ao.zip
curl -o nld-aa/nld-aa-dir-ap.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ap.zip
```



`NLD_NAO` (41gi files)
```
# Download NLD-NAO
curl -o nld-nao/nld-nao-dir-aa.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aa.zip
curl -o nld-nao/nld-nao-dir-ab.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ab.zip
curl -o nld-nao/nld-nao-dir-ac.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ac.zip
curl -o nld-nao/nld-nao-dir-ad.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ad.zip
curl -o nld-nao/nld-nao-dir-ae.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ae.zip
curl -o nld-nao/nld-nao-dir-af.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-af.zip
curl -o nld-nao/nld-nao-dir-ag.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ag.zip
curl -o nld-nao/nld-nao-dir-ah.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ah.zip
curl -o nld-nao/nld-nao-dir-ai.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ai.zip
curl -o nld-nao/nld-nao-dir-aj.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aj.zip
curl -o nld-nao/nld-nao-dir-ak.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ak.zip
curl -o nld-nao/nld-nao-dir-al.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-al.zip
curl -o nld-nao/nld-nao-dir-am.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-am.zip
curl -o nld-nao/nld-nao-dir-an.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-an.zip
curl -o nld-nao/nld-nao-dir-ao.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ao.zip
curl -o nld-nao/nld-nao-dir-ap.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ap.zip
curl -o nld-nao/nld-nao-dir-aq.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aq.zip
curl -o nld-nao/nld-nao-dir-ar.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ar.zip
curl -o nld-nao/nld-nao-dir-as.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-as.zip
curl -o nld-nao/nld-nao-dir-at.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-at.zip
curl -o nld-nao/nld-nao-dir-au.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-au.zip
curl -o nld-nao/nld-nao-dir-av.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-av.zip
curl -o nld-nao/nld-nao-dir-aw.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aw.zip
curl -o nld-nao/nld-nao-dir-ax.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ax.zip
curl -o nld-nao/nld-nao-dir-ay.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ay.zip
curl -o nld-nao/nld-nao-dir-az.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-az.zip
curl -o nld-nao/nld-nao-dir-ba.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ba.zip
curl -o nld-nao/nld-nao-dir-bb.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bb.zip
curl -o nld-nao/nld-nao-dir-bc.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bc.zip
curl -o nld-nao/nld-nao-dir-bd.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bd.zip
curl -o nld-nao/nld-nao-dir-be.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-be.zip
curl -o nld-nao/nld-nao-dir-bf.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bf.zip
curl -o nld-nao/nld-nao-dir-bg.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bg.zip
curl -o nld-nao/nld-nao-dir-bh.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bh.zip
curl -o nld-nao/nld-nao-dir-bi.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bi.zip
curl -o nld-nao/nld-nao-dir-bj.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bj.zip
curl -o nld-nao/nld-nao-dir-bk.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bk.zip
curl -o nld-nao/nld-nao-dir-bl.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bl.zip
curl -o nld-nao/nld-nao-dir-bm.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bm.zip
curl -o nld-nao/nld-nao-dir-bn.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bn.zip
curl -o nld-nao/nld-nao_xlogfiles.zip https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao_xlogfiles.zip
```


### Reconstructing the Dataset

Unzip the files in the standard way, with separate directories for `NLD-AA`, and `NLD-NAO`. 



```bash
# for NLD-AA
# will give you an nle_data directory at /path/to/dir/nld-aa-dir/nld-aa/nle_data/
$ unzip /path/to/nld-aa/nld-aa-dir-aa.zip  -d /path/to/dir
$ unzip /path/to/nld-aa/nld-aa-dir-ab.zip  -d /path/to/dir
$ unzip /path/to/nld-aa/nld-aa-dir-ac.zip  -d /path/to/dir
...


# for NLD-NAO - don'f forget xlogfiles
$ unzip /path/to/nld-xlogfiles.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-aa.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-ab.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-ac.zip -d /path/to/nld-nao
...
```

## Using the Dataset

The code needed to use the dataset will be distributed in `NLE v0.9.0`. For now it can be found on the `main` branch of [NLE](https://github.com/facebookresearch/nle). You can follow the instructions to install [there](https://github.com/facebookresearch/nle), or try the below.

```
# With pip:
pip install git+https://github.com/facebookresearch/nle.git@main

# From source:
git clone --recursive https://github.com/facebookresearch/nle.git
cd nle && pip install -e .
```

Once this is installed, you simply need to load the `nld` folders (once) which will create a small local sqlite3 database, and then you can use the dataset.

```python
import nle.dataset as nld

if not nld.db.exists():
    nld.db.create()
    # NB: Different methods are used for data based on NLE and data from NAO.
    nld.add_nledata_directory("/path/to/nld-aa", "nld-aa-v0")
    nld.add_altorg_directory("/path/to/nld-nao", "nld-nao-v0")

dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=128, ...)
for i, mb in enumerate(dataset):
    foo(mb) # etc...
```

for more instructions on usage see the accompanying tutorial notebook in this repo.


## Replicating Experiments

Arguments used for replicating different experiments (from authors) can be found in `experiment_code/runs.sh`. To examine if setup is done correctly and everything works I recommend running APPO from scratch with Human Monk.

```bash
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal'
```

### Running experiment with conda (or inside singularity shell)

If you want to run this experiment locally (conda) you can run `experiment_code/train.sh` or simply:

```bash
export BROKER_IP=0.0.0.0
export BROKER_PORT=4431

python -m moolib.broker &

python -m hackrl.experiment connect=$BROKER_IP:$BROKER_PORT exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal' group='monk-APPO'
```

### Running debugging session

If you want to start a debugging session (vscode) I recommend doing it with debugpy.

```bash
export BROKER_IP=0.0.0.0
export BROKER_PORT=4431

python -m moolib.broker &

python -m debugpy --wait-for-client --listen 5678 ./hackrl/experiment.py connect=$BROKER_IP:$BROKER_PORT exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal' group='monk-APPO'
```

And to connect your client you need to add to launch.json 
```bash
    {
        "name": "Python: Attach",
        "type": "python",
        "request": "attach",
        "connect": {
            "host": "localhost",
            "port": 5678
        },
        "justMyCode": false,
    }
```

### Using sbatch command

I recommend using pure sbatch and singularity image for running experiments. 
Example:

```bash
sbatch run_ares.sh
```

## Troubleshooting

If you are having issues loading the dataset, ensure that the directory structure is as laid out in the docstrings to the `add_*_directory` functions.

``` python
help(nld.add_nledata_directory) # will print docstring 
```

Or if you need to get in touch please send an email to eric *DOT* hambro _AT_ gmail *DOT* com


