# Use Ubuntu base and manually install CUDA
FROM ubuntu:20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA keyring and repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install core ML packages with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    numpy==1.26.4

# Install HuggingFace ecosystem
RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    tokenizers==0.21.0

# Install datasets and evaluation tools
RUN pip install --no-cache-dir \
    datasets==3.1.0 \
    evaluate==0.4.3 \
    scikit-learn==1.6.0

# Install additional ML utilities
RUN pip install --no-cache-dir \
    accelerate==1.2.1 \
    pandas==2.2.3 \
    requests==2.32.4 \
    tqdm==4.67.1

# Install LLM clients
RUN pip install --no-cache-dir openai
RUN pip install --no-cache-dir anthropic

# Install Docker Python client
RUN pip install --no-cache-dir docker

# Install Aider for code fixing
RUN pip install --no-cache-dir aider-chat

# Set working directory
WORKDIR /workspace

# Set up git config (required by aider)
RUN git config --global user.email "docker@example.com" && \
    git config --global user.name "Docker Container"

# Set environment variables to reduce warnings
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONWARNINGS=ignore
