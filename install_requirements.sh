#!/bin/bash
# run the script after activating the created conda environment
set -e

echo "Step 1: Installing PyTorch packages..."
pip install -r requirements_torch.txt

echo "Step 2: Installing other dependencies..."
pip install -r requirements.txt

echo "Installation completed successfully!"
