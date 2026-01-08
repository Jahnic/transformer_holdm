#!/bin/bash

# Setup script for Transformer Chess project

set -e

echo "=== Transformer Chess Setup ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.9" | bc) -eq 1 ]]; then
    echo "Warning: Python 3.9+ recommended"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install PyTorch with MPS support (for Apple Silicon)
echo ""
echo "Ensuring PyTorch has MPS (Metal) support for Apple Silicon..."
pip install --upgrade torch

# Verify MPS availability
echo ""
echo "Checking MPS availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✓ Your Mac can use GPU acceleration!')
else:
    print('Note: MPS not available, will use CPU (still fast on M4)')
"

# Create data directories
mkdir -p data/raw data/processed

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Download chess data: cd data && bash download.sh"
echo "3. Or create synthetic data: python src/data.py"
echo "4. Test the dataloader: python src/dataset.py"
echo ""
