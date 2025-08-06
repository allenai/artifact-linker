# Use NVIDIA CUDA runtime as base
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install core ML packages with CUDA support
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
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
