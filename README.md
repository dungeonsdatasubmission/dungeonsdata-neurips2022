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

The dataset is currently hosted on WeTransfer with open access for all, and will remain there for the duration of the review period. It will eventually move to its own dedicated hosting site, which is in the process of being set up. For the time being, `NLD-AA` is one file, while `NLD-NAO` is in 5 parts (4 ttyrec zips + the xlogfiles).


### Download Links


`NLD-AA` (1 file)
- [`nld-aa.zip`](https://we.tl/t-wwN4lD7Hqn) 


`NLD_NAO` (5 files)
- [`nld-nao_part1.zip`](https://we.tl/t-XQe15aXAes)
- [`nld-nao_part2.zip`](https://we.tl/t-YRHHAb9gTe)
- [`nld-nao_part3.zip`](https://we.tl/t-XB0iundCAU)
- [`nld-nao_part4.zip`](https://we.tl/t-pkWlT0yTFK)
- [`nld-nao_xlogfiles.zip`](https://we.tl/t-vy7IAGohCu)

### Reconstructing the Dataset

Unzip the files in the standard way, with separate directories for `NLD-AA`, and `NLD-NAO`. 


```bash
$ unzip /path/to/nld-aa.zip 

$ unzip /path/to/nld-xlogfiles.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part1.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part2.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part3.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part4.zip -d /path/to/nld-nao
```


- NB: `NLD-AA` is already a single directory, so will unzip to one directory already,
where as all the `NLD-NAO` files should be zipped to one directory.

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

Code with a `README.md` on how to replicate experiments is available in the `experiment_code` directory.  This code was developed for use on an internal cluster, and will be tidied up and open sourced in NLE upon full release of the dataset.

## Troubleshooting

If you are having issues loading the dataset, ensure that the directory structure is as laid out in the docstrings to the `add_*_directory` functions.

``` python
help(nld.add_nledata_directory) # will print docstring 
```

Or if you need to get in touch email dungeons.data.submission@gmail.com


