#!/bin/bash
# setup_environment.sh - Complete environment setup for GR00T Dreams

set -e  # Exit on any error

echo "🤖 GR00T Dreams Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
    print_status "CUDA detected: $CUDA_VERSION"
else
    print_warning "CUDA not detected. Installing CPU-only version."
    print_info "For GPU acceleration, install CUDA drivers first."
fi

# Remove existing environment if it exists
ENV_NAME="groot-dreams"
if conda env list | grep -q "$ENV_NAME"; then
    print_warning "Environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create conda environment from file
print_info "Creating conda environment from environment.yml..."
conda env create -f environment.yml

print_status "Conda environment '$ENV_NAME' created successfully!"

# Activate environment and install the package in development mode
print_info "Activating environment and installing GR00T Dreams package..."

# Use conda run to execute commands in the environment
conda run -n "$ENV_NAME" pip install -e .

print_status "GR00T Dreams package installed in development mode!"

# Verify installation
print_info "Verifying installation..."
conda run -n "$ENV_NAME" python -c "
try:
    import gr00t
    print('✅ GR00T package imported successfully')
    print(f'   Version: {gr00t.__version__ if hasattr(gr00t, \"__version__\") else \"dev\"}')
except ImportError as e:
    print(f'❌ Failed to import GR00T: {e}')
    exit(1)

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'   CUDA available: {torch.cuda.get_device_name(0)}')
        print(f'   CUDA version: {torch.version.cuda}')
    else:
        print('   CUDA not available (CPU-only)')
except ImportError:
    print('❌ PyTorch not installed')
    exit(1)

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('⚠️  Transformers not installed')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError:
    print('❌ OpenCV not installed')
"

if [ $? -eq 0 ]; then
    print_status "Installation verification completed successfully!"
else
    print_error "Installation verification failed!"
    exit 1
fi

# Create activation script
cat > activate_groot.sh << 'EOF'
#!/bin/bash
# Quick activation script for GR00T Dreams environment

echo "🤖 Activating GR00T Dreams environment..."
conda activate groot-dreams

echo "Environment activated! 🚀"
echo ""
echo "Quick commands:"
echo "  🔍 Test dataset loading:    python paper_return_examples.py <dataset_path>"
echo "  📊 Run preprocessing:       bash IDM_dump/scripts/preprocess/paper_return.sh"  
echo "  🏋️  Start fine-tuning:       bash IDM_dump/scripts/finetune/paper_return.sh"
echo "  📚 Open Jupyter Lab:        jupyter lab"
echo ""
echo "Next steps:"
echo "  1. Download your dataset: huggingface-cli download Hafnium49/paper_return --local-dir ./paper_return_dataset"
echo "  2. Follow the integration guide: cat PAPER_RETURN_INTEGRATION_GUIDE.md"
EOF

chmod +x activate_groot.sh

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Environment: $ENV_NAME"
echo "Activation: conda activate $ENV_NAME"
echo "Quick start: ./activate_groot.sh"
echo ""
print_status "Your GR00T Dreams environment is ready!"
echo ""
echo "Next steps:"
echo "  1. 🔑 Login to Hugging Face (if using private datasets):"
echo "     conda activate $ENV_NAME && huggingface-cli login"
echo ""  
echo "  2. 📥 Download your dataset:"
echo "     huggingface-cli download Hafnium49/paper_return --local-dir ./paper_return_dataset"
echo ""
echo "  3. 🧪 Test the setup:"
echo "     python paper_return_examples.py ./paper_return_dataset"
echo ""
echo "  4. 📖 Read the integration guide:"
echo "     cat PAPER_RETURN_INTEGRATION_GUIDE.md"
echo ""
print_info "Happy robot learning! 🤖🚀"