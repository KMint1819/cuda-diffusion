FROM jnativ/ece408_minidnn_docker_sp21:latest
# FROM raiproject/pumps2018:amd64-cuda100

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update && \
    apt -y upgrade

COPY requirements.txt /tmp
COPY nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb /tmp
RUN apt install -y /tmp/nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb

RUN apt install -y \
    python3.8 \
    python3-pip \
    python3.8-dev

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install \
    Cython \ 
    triton==2.0.0

RUN python3.8 -m pip install -r /tmp/requirements.txt
WORKDIR /workspace
