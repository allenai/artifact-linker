#!/bin/bash

# Build script for artifact-linker-verification Docker image

echo "🐳 Building artifact-linker-verification Docker image..."

# Clean up old containers first to free space
echo "🧹 Cleaning up old containers..."
docker container prune -f > /dev/null 2>&1

# Build the image
docker build --no-cache -t artifact-linker-verification:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo "📋 Image details:"
    docker images artifact-linker-verification:latest

    echo ""
    echo "🧪 Testing the image..."
    echo "  📦 Python version:"
    docker run --rm artifact-linker-verification:latest python --version

    echo "  📚 Key packages:"
    docker run --rm artifact-linker-verification:latest python -c "
import torch, transformers, datasets
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  Datasets: {datasets.__version__}')
"

    echo "  🔇 Warning suppression test:"
    docker run --rm artifact-linker-verification:latest python -c "
import os
print(f'  TF_CPP_MIN_LOG_LEVEL: {os.environ.get(\"TF_CPP_MIN_LOG_LEVEL\")}')
print(f'  PYTHONWARNINGS: {os.environ.get(\"PYTHONWARNINGS\")}')
"

    echo ""
    echo "🎯 Image is ready for use!"
    echo "💡 Usage examples:"
    echo "   # Basic usage:"
    echo "   python examples/hf_auto_eval.py --llm-model gpt-4o"
    echo ""
    echo "   # With custom settings:"
    echo "   python examples/hf_auto_eval.py --memory-limit 8g --output-dir results"
    echo ""
    echo "   # With GPU support (if available):"
    echo "   python examples/hf_auto_eval.py --memory-limit 16g"
else
    echo "❌ Docker build failed!"
    exit 1
fi
