#!/bin/bash
# Install cosmos-predict2 in the current conda environment

set -e

echo "=== Installing Cosmos Predict2 ==="

# Navigate to cosmos-predict2 directory
cd /home/hafnium/cosmos-predict2

# Check Python version
echo "Current Python version:"
python --version

# Install uv package manager if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Option 1: Install cosmos-predict2 package in editable mode
echo "Installing cosmos-predict2 in editable mode..."
pip install -e .

# Install additional dependencies if needed
echo "Installing additional dependencies..."
pip install opencv-python pillow

echo "✅ Cosmos Predict2 installation complete!"

# Test import
echo -e "\nTesting import..."
python -c "
try:
    import imaginaire
    import cosmos_predict2
    print('✅ Successfully imported imaginaire and cosmos_predict2')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

cd -