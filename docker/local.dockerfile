FROM kmint1819/gten:rai

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

COPY nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb /tmp
RUN apt install -y /tmp/nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb

WORKDIR /workspace
