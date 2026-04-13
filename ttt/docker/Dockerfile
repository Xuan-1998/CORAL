# Build on top of a SLIME base image which already includes:
# SGLang, Megatron-LM, flash-attn, TransformerEngine, apex, DeepEP, Ray, uv, etc.
#
# Build from the CORAL repo root:
#   docker build -t coral-ttt -f ttt/docker/Dockerfile .
#
# Override the base image:
#   docker build --build-arg BASE_IMAGE=slimerl/slime:latest -t coral-ttt -f ttt/docker/Dockerfile .
#
# Run (requires --ulimit nofile for Ray):
#   docker run --gpus all --ipc=host --network host \
#     --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
#     -e CORAL_TASK_YAML=/app/examples/circle_packing/task.yaml \
#     -e HF_CKPT=/path/to/model \
#     coral-ttt:latest

ARG BASE_IMAGE=slimerl/slime:latest

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Save base image torch version before any pip installs can upgrade it
RUN python3 -c "import torch; print(torch.__version__)" > /tmp/torch_ver.txt 2>/dev/null || true \
    && python3 -c "import transformers; print(transformers.__version__)" > /tmp/tf_ver.txt 2>/dev/null || true

# =========================
# Install vanilla Megatron-LM
# =========================
RUN pip install pybind11
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
RUN git clone https://github.com/NVIDIA/Megatron-LM.git /opt/megatron-lm && \
    cd /opt/megatron-lm && git checkout ${MEGATRON_COMMIT} && \
    pip install -e . --no-deps --no-build-isolation
ENV MEGATRON_ROOT=/opt/megatron-lm

# =========================
# Install Node.js and opencode (agent runtime)
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g opencode-ai@latest

# =========================
# Install SLIME and CORAL into system Python
# =========================
WORKDIR /app

COPY ttt/slime/ ttt/slime/
RUN cd ttt/slime && pip install -e . --no-deps --no-build-isolation

COPY pyproject.toml README* ./
COPY coral/ coral/
COPY ttt/ ttt/
# Pin torch to base image version to prevent litellm/accelerate/peft from upgrading it,
# which would break pre-compiled CUDA extensions (sgl_kernel, flash_attn, torchvision).
RUN TORCH_VER=$(cat /tmp/torch_ver.txt) \
    && pip install hatchling hatch-vcs editables \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e . --no-deps --no-build-isolation \
    && pip install --ignore-installed "litellm[proxy]" httpx fastapi uvicorn pylatexenc accelerate peft \
        "torch==${TORCH_VER}" "huggingface-hub<1.0,>=0.34.0" \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        "numpy<2" \
    && TF_VER=$(cat /tmp/tf_ver.txt) \
    && if [ -n "$TF_VER" ]; then pip install "transformers==$TF_VER" --no-deps 2>/dev/null || true; fi

# =========================
# Entrypoint
# =========================
ENV CORAL_IN_DOCKER=1
# PYTHONPATH must be set globally so Ray workers inherit it
ENV PYTHONPATH="/app/ttt:/app/ttt/slime:/opt/megatron-lm"
COPY ttt/docker/entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
