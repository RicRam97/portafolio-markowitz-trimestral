#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Starting build process..."

# Upgrade pip to ensure smooth binary installations (Pandas/Scipy)
python -m pip install --upgrade pip wheel

# Install Python requirements
pip install -r requirements.txt

echo "Build complete."
