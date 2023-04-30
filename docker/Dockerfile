FROM raiproject/pumps2018:amd64-cuda100

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

COPY requirements.txt /tmp

RUN apt update && \
    apt -y upgrade

RUN apt install -y \
    python3.8 \
    python3-pip \
    python3.8-dev

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install \
    Cython \ 
    triton==2.0.0

RUN python3.8 -m pip install -r /tmp/requirements.txt
