FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    build-essential \
    git \
    sudo \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN python3.10 -m pip install --upgrade pip setuptools wheel virtualenv

RUN python3.10 -m venv /opt/venvs/py310_env

RUN /opt/venvs/py310_env/bin/pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY . /app

RUN python3.10 -m pip install -e .
RUN python3.10 -m pip install -r requirements_isaacgym.txt

RUN /opt/venvs/py310_env/bin/pip install -e pyroki-main
RUN /opt/venvs/py310_env/bin/python -m pip install --upgrade "jax[cuda]" 

# Expose all GPUs inside the container by default
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Default command to open bash
CMD ["bash"]
