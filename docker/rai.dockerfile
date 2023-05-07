FROM jnativ/ece408_minidnn_docker_sp21:latest
# FROM raiproject/pumps2018:amd64-cuda100

RUN wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth -P /home

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update && \
    apt -y upgrade

COPY requirements.txt /tmp

RUN apt install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install \
    Cython \ 
    triton==2.0.0

RUN python3.8 -m pip install -r /tmp/requirements.txt
WORKDIR /build

RUN apt install -y \
    libsm6 \
    libxext6 \
    libxrender-dev
COPY pre.py /tmp
RUN python3.8 /tmp/pre.py
