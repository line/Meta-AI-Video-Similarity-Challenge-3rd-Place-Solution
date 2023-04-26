FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu18.04

RUN : \
    && apt-get update \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN : \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm /etc/apt/sources.list.d/cuda.list
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.9-venv python3.9-dev \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git libmagic-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /app

RUN pip3 install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st --no-deps

ENV PYTHONPATH=/app/vsc2022/
