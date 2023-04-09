# -*- mode: dockerfile -*-

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        build-essential \
        cmake \
        curl \
        git \
        ninja-build \
        wget \
        make

RUN apt-get install -yq autoconf libtool pkg-config
RUN apt-get install -yq libbz2-dev
RUN apt-get install -yq libgl1-mesa-dev libgl1-mesa-glx

WORKDIR /opt/conda_setup

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x miniconda.sh && \
     ./miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip

RUN conda update -n base -c defaults conda
RUN conda install cmake flex bison lit
RUN conda install pybind11 -c conda-forge
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN pip install -q tqdm debugpy

WORKDIR /opt/

# install moolib
RUN git clone --recursive https://github.com/facebookresearch/moolib.git
WORKDIR /opt/moolib
RUN pip install -r examples/requirements.txt
RUN pip install .

# install nle
RUN pip install nle

# copy current directory
WORKDIR /opt/dungeons
COPY . /opt/dungeons/

# install dungeons and data
RUN pip install -r requirements.txt 
RUN pybind11_INCLUDE_DIR=/opt/conda/share/cmake/pybind11
WORKDIR /opt/dungeons/render_utils
RUN pip install -e . 
WORKDIR /opt/dungeons
RUN pip install -e .

# test moolib and nle
RUN python -c "import moolib"
RUN python -c 'import gym; import nle; env = gym.make("NetHackScore-v0"); env.reset()'

RUN    apt-get clean
RUN    rm -rf /var/lib/apt/lists/*
RUN    conda clean --all
RUN    conda info

CMD ["bash", "-c", "python examples/a2c.py"]

# Docker commands:
#   docker rm dungeons -v
#   docker build -t dungeons -f Dockerfile .
#   docker run --gpus all --rm --name dungeons dungeons
# or
#   docker run --gpus all -it --entrypoint /bin/bash dungeons