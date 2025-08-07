# Use NVIDIA CUDA runtime base image with Python
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

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
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install core ML packages with CUDA support (use latest stable versions)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
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
