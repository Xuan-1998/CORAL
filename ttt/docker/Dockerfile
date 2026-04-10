# vllm018
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    build-essential \
    libibverbs-dev \
    libnuma-dev \
    librdmacm-dev \
    numactl \
    software-properties-common \
    vim && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv early so all subsequent Python installs use it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129

RUN uv pip install --system pybind11 wheel

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --system nvidia-mathdx ninja

# apex requires --config-settings which uv pip doesn't support; use pip
RUN MAX_JOBS=256 pip install -v --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

RUN export NVTE_FRAMEWORK=pytorch && \
    MAX_JOBS=256 \
    NVTE_BUILD_THREADS_PER_JOB=4 \
    uv pip install --system --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.12

RUN uv pip install --system codetiming mathruler pylatexenc qwen_vl_utils cachetools pytest-asyncio

RUN export FLASH_ATTENTION_FORCE_BUILD="TRUE" && MAX_JOBS=32 uv pip install --system --no-build-isolation flash_attn==2.8.3

RUN NSIGHT_VERSION=2025.6.1_2025.6.1.190-1_$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-${NSIGHT_VERSION}.deb && \
    apt-get update && apt-get install -y libxcb-cursor0 && \
    apt-get install -y ./nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /usr/local/cuda/bin/nsys && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys  /usr/local/cuda/bin/nsys && \
    rm -rf /usr/local/cuda/bin/nsys-ui && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys-ui /usr/local/cuda/bin/nsys-ui && \
    rm nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /var/lib/apt/lists/*

# =========================
# Install DeepEP
# =========================
WORKDIR /home/dpsk_a2a
RUN git clone -b v2.5.1 https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    make prefix=/usr/local lib_install && \
    cd .. && rm -rf gdrcopy

ENV GDRCOPY_HOME=/usr/local

RUN git clone -b hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    export NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem && \
    export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH" && \
    export PATH="${NVSHMEM_DIR}/bin:$PATH" && \
    cd ${NVSHMEM_DIR}/lib && \
    ln -sf libnvshmem_host.so.3 libnvshmem_host.so && \
    cd /home/dpsk_a2a/DeepEP && \
    export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:$CPATH && \
    python setup.py install

RUN uv pip install --system vllm==0.18.0

RUN uv pip install --system --no-deps trl==0.27.0

RUN uv pip install --system nvtx matplotlib liger_kernel

RUN uv pip install --system -U git+https://github.com/ISEEKYAN/mbridge.git@641a5a0

RUN uv pip install --system --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0

RUN uv pip install --system transformers==5.3.0

RUN uv pip install --system git+https://github.com/volcengine/verl.git@v0.7.1 && \
    uv pip uninstall --system verl

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --allow-downgrades --allow-change-held-packages \
    libcudnn9-cuda-12=9.16.0.29-1 \
    libcudnn9-dev-cuda-12=9.16.0.29-1 \
    libcudnn9-headers-cuda-12=9.16.0.29-1 && \
    rm -rf /var/lib/apt/lists/*

# =========================
# Install Node.js, opencode, and CORAL
# =========================
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g opencode-ai@latest

# Install CORAL with ttt extra into the system Python.
# --no-build-isolation reuses torch/flash-attn already installed above.
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT="/usr"
COPY pyproject.toml uv.lock* README* ./
COPY coral/ coral/
COPY ttt/ ttt/
RUN uv sync --no-dev --extra ttt --no-build-isolation

# Entrypoint copies mounted credentials, then runs the ttt trainer
ENV CORAL_IN_DOCKER=1
COPY ttt/docker/entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
