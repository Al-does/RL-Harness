FROM ubuntu:24.04

LABEL org.opencontainers.image.source="https://github.com/Al-does/RL-Harness" \
      org.opencontainers.image.description="Pinned RL Harness environment for RunPod Pods"

ARG RAY_VERSION=2.56.0
ARG TORCH_VERSION=2.12.1
ARG GYMNASIUM_VERSION=1.2.2
ARG BOTO3_VERSION=1.43.51
ARG MLFLOW_VERSION=3.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    RUNPOD_VENV_DIR=/opt/venv

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-server tini \
    && rm -f /etc/ssh/ssh_host_* \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv python install 3.13 \
    && /root/.local/bin/uv venv --python 3.13 /opt/venv

# The pinned Linux torch wheel brings its CUDA 13 runtime libraries. Starting
# from Ubuntu avoids duplicating the CUDA/torch stack from a PyTorch base image.
# Dependencies stay above source COPY so runner edits preserve this layer.
RUN /root/.local/bin/uv pip install --no-cache --python /opt/venv/bin/python \
        "ray[rllib]==${RAY_VERSION}" \
        "torch==${TORCH_VERSION}" \
        "gymnasium==${GYMNASIUM_VERSION}" \
        "matplotlib==3.11.1" \
        "scipy==1.18.0" \
        "boto3==${BOTO3_VERSION}" \
        "mlflow-skinny==${MLFLOW_VERSION}"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Experiment and harness source are not baked. The entrypoint clones both refs
# at startup, then editable-installs them without resolving dependencies.
COPY devops/runpod/pods/container_entrypoint.py /opt/runpod/container_entrypoint.py

EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/opt/venv/bin/python", "/opt/runpod/container_entrypoint.py"]
