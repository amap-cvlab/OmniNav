FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget software-properties-common \
    libegl1 libgl1 libglib2.0-0 libjpeg-dev \
    libsm6 libx11-6 libxext6 libxrender1 ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3.9 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && ln -s /usr/bin/python3.9 /usr/local/bin/python

COPY requirements-docker.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone --recursive --depth 1 --branch v0.1.7 \
    https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim \
    && cd /opt/habitat-sim \
    && pip install -r requirements.txt \
    && python setup.py install --headless

RUN git clone --recursive --depth 1 --branch v0.1.7 \
    https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
    && cd /opt/habitat-lab \
    && pip install -r habitat-baselines/habitat_baselines/rl/requirements.txt \
    && pip install -r habitat-baselines/habitat_baselines/rl/ddppo/requirements.txt \
    && pip install -e . \
    && pip install -e habitat-baselines

WORKDIR /workspace/OmniNav
COPY . .

RUN pip uninstall -y transformers || true \
    && pip install -e ./train_code/transformers-main \
    && pip install flash-attn==2.7.4.post1 --no-build-isolation

CMD ["/bin/bash"]
