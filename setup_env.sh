#!/bin/bash
# QuTLASS Environment Setup Script
#
# This script sets up a clean conda environment for QuTLASS development and testing

set -e

echo "=========================================="
echo "QuTLASS Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found!"
    echo "   Install miniconda or miniforge first"
    exit 1
fi

# Remove old environment if it exists
if conda env list | grep -q "qutlass"; then
    echo "Removing existing qutlass environment..."
    conda env remove -n qutlass -y
fi

# Create new environment
echo ""
echo "Creating qutlass environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "✅ Environment created successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   conda activate qutlass"
echo ""
echo "2. Install QuTLASS:"
echo "   pip install --no-build-isolation -e ."
echo ""
echo "3. (Optional) Install FlashInfer from local fork:"
echo "   cd ../flashinfer && pip install -e ."
echo ""
echo "4. Run tests:"
echo "   pytest tests/mxfp4_test.py -v -k 'not flashinfer'"
echo ""
echo "5. Fix cuDNN conflict if needed:"
echo "   pip uninstall nvidia-cudnn-cu12 -y"
echo "   export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo ""
