# Use Python base image with CUDA 12.8 for Blackwell GPU support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

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
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch nightly with CUDA 12.8 support (for Blackwell GPUs - compute capability 12.0)
RUN pip install --no-cache-dir --pre \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install HuggingFace ecosystem
RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    tokenizers==0.21.0 \
    datasets==3.1.0 \
    huggingface_hub

# Install evaluation metrics packages
RUN pip install --no-cache-dir \
    evaluate==0.4.3 \
    rouge-score \
    sacrebleu \
    bert-score \
    nltk \
    py-rouge \
    rouge \
    mauve-text \
    unbabel-comet

# Install scikit-learn and ML utilities
RUN pip install --no-cache-dir \
    scikit-learn==1.6.0 \
    scipy \
    numpy

# Install data processing packages
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    pyarrow \
    openpyxl \
    xlrd

# Install JSON and YAML utilities
RUN pip install --no-cache-dir \
    jsonlines \
    orjson \
    ujson \
    pyyaml \
    toml

# Install HTTP/API packages
RUN pip install --no-cache-dir \
    requests==2.32.4 \
    aiohttp \
    httpx

# Install LLM clients
RUN pip install --no-cache-dir \
    openai \
    anthropic \
    tiktoken

# Install additional ML utilities
RUN pip install --no-cache-dir \
    accelerate==1.2.1 \
    tqdm==4.67.1 \
    matplotlib \
    seaborn \
    pillow

# Install NLP utilities
RUN pip install --no-cache-dir \
    sentencepiece \
    protobuf \
    spacy

# Install misc utilities
RUN pip install --no-cache-dir \
    regex \
    filelock \
    packaging

# Download NLTK data for evaluation
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Set working directory
WORKDIR /workspace

# Set up git config
RUN git config --global user.email "docker@example.com" && \
    git config --global user.name "Docker Container"

# Set environment variables to reduce warnings
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONWARNINGS=ignore
ENV HF_HUB_DISABLE_TELEMETRY=1

# Default command
CMD ["python"]
