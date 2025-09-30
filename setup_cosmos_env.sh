#!/bin/bash
# Setup a dedicated Cosmos Predict2 conda environment

set -e

echo "=== Setting up Cosmos Predict2 Environment ==="

# Create conda environment with Python 3.10
echo "1. Creating cosmos conda environment with Python 3.10..."
conda create -n cosmos python=3.10 -y

# Activate and setup the environment
echo -e "\n2. Installing Cosmos Predict2 dependencies..."

# Install PyTorch with CUDA 12.1
conda run -n cosmos pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install uv for faster package management
conda run -n cosmos pip install uv

# Navigate to cosmos-predict2 directory
cd /home/hafnium/cosmos-predict2

# Install cosmos-predict2 package
echo -e "\n3. Installing Cosmos Predict2..."
conda run -n cosmos pip install -U "cosmos-predict2[cu126]" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple

# Install flash-attn
echo -e "\n4. Installing flash-attn..."
conda run -n cosmos pip install flash-attn --no-build-isolation

# Install additional dependencies
echo -e "\n5. Installing additional dependencies..."
conda run -n cosmos pip install -e .

# Test the installation
echo -e "\n6. Testing installation..."
conda run -n cosmos python -c "
import sys
sys.path.insert(0, '/home/hafnium/cosmos-predict2')
try:
    import torch
    import imaginaire
    import cosmos_predict2
    print('✅ PyTorch version:', torch.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
    print('✅ Cosmos Predict2 imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo -e "\n✅ Cosmos environment setup complete!"
echo "To activate: conda activate cosmos"
echo "To run inference: conda run -n cosmos python <script.py>"