FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.10 \
    python3.8-dev \
    python3.10-dev \
    python3-pip \
    curl \
    ca-certificates \
    build-essential \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3.8 python3.8 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3.10 python3.10 /usr/bin/python3.10 1

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN python3.8 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY . /app

WORKDIR /app

RUN python3.8 -m pip install --no-cache-dir -e .
RUN python3.8 -m pip install -r requirements_isaacgym.txt

# Expose all GPUs inside the container by default
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
