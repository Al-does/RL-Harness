# Reuse the exact Python 3.13 / Ray 2.56.0 / Torch 2.12.1 /
# Gymnasium 1.2.2 / CUDA 13 environment from the published Pods image.
# Cold-start cost is dominated by the Torch CUDA wheel layer in that base;
# this image intentionally adds only the worker SDK + handler. See
# devops/runpod/execution/image_coldstart.md.
FROM ghcr.io/al-does/rl-harness-runpod@sha256:1257ac0a0f2b57b80022849a96fa1d9a5bfffff69fabbf233b3cf45dc665fb3c

LABEL org.opencontainers.image.source="https://github.com/Al-does/RL-Harness" \
      org.opencontainers.image.description="Disposable RL Harness RunPod Serverless worker"

# The SDK is worker-only and intentionally not a core rl-harness dependency.
# Resolve the latest Python-3.13-compatible release when publishing the image;
# the resulting worker image itself must then be launched by immutable digest.
RUN /root/.local/bin/uv pip install --no-cache --upgrade \
      --python /opt/venv/bin/python runpod

COPY devops/serverless/handler.py /opt/runpod/serverless_handler.py
# Shared phased-execution helpers used by the baked handler for durability and
# results publication. Keep this COPY after dependency install so edits stay cheap.
COPY devops /opt/rl-harness-devops/devops
ENV PYTHONPATH=/opt/rl-harness-devops:${PYTHONPATH}

ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/opt/venv/bin/python", "-u", "/opt/runpod/serverless_handler.py"]
