FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# openssh-client is required for devops.vast.provision readiness probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    libgomp1 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pre-install the project Python so `uv sync` starts faster on agent boot.
RUN uv python install 3.13
