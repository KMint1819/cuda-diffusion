FROM raiproject/pumps2018:amd64-cuda100

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update && \
    apt -y upgrade

RUN apt install -y \
    python3.8

RUN python3.8 -m pip install --upgrade pip

