FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/radar
WORKDIR /opt/radar
RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache . \
    && rm -rf ./build ./*.egg-info


ENV CUDA_VISIBLE_DEVICES="0"
ENTRYPOINT ["/usr/local/bin/radar"]
