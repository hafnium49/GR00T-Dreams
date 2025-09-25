#!/bin/bash

# Quick activation script for GR00T Dreams conda environment
# Usage: source activate_groot.sh

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 GR00T Dreams Environment Activator${NC}"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}⚠️  Conda not found. Please install Miniconda or Anaconda first.${NC}"
    return 1
fi

# Check if groot-dreams environment exists
if conda env list | grep -q "groot-dreams"; then
    echo -e "${GREEN}✅ Activating groot-dreams environment...${NC}"
    conda activate groot-dreams
    
    # Show environment info
    echo -e "${BLUE}📋 Environment Information:${NC}"
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    echo "Working directory: $(pwd)"
    
    # Check if we're in the GR00T-Dreams directory
    if [[ $(basename "$PWD") == "GR00T-Dreams" ]]; then
        echo -e "${GREEN}✅ You're in the GR00T Dreams directory!${NC}"
        echo -e "${BLUE}💡 Quick start commands:${NC}"
        echo "  python load_dataset.py                     # Load and inspect datasets"
        echo "  python paper_return_examples.py <dataset>  # Work with your dataset"
        echo "  jupyter lab getting_started/               # Open tutorials"
    else
        echo -e "${YELLOW}💡 Navigate to your GR00T-Dreams directory to get started${NC}"
    fi
    
else
    echo -e "${YELLOW}❌ Environment 'groot-dreams' not found.${NC}"
    echo "Create it with: ./setup_environment.sh"
    return 1
fi