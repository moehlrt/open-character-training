#!/bin/bash

# DPO Dataset Generation Launcher
# This script sets up the environment on a fresh GPU instance (e.g., Lambda Labs)
# and launches the DPO dataset creation process.

# 1) Example launch (simplest)
# bash run_dpo.sh
# 2) Example launch in a screen session (cause the run may take a while):
# screen -L -Logfile dpo_run.log -S dpo_session bash run_dpo.sh

# 1. Configuration
# ------------------------------------------------------------------------------
# Name of your main Python script (must be in the same directory or relative path)
PYTHON_SCRIPT="scripts/create_dpo_dataset.py"

# Hugging Face Token for accessing the models
# Ensure the HF_TOKEN environment variable is set before running this script.
# You can set it via: export HF_TOKEN="hf_YourToken..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "Models require authentication. The script might fail if not logged in."
fi

# 2. Install UV (Modern Python Package Manager)
# ------------------------------------------------------------------------------
echo "--- Checking/Installing uv ---"
# If uv is not found, install it automatically
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "uv is already installed."
fi

# 3. Install Dependencies
# ------------------------------------------------------------------------------
echo "--- Syncing Dependencies (GPU Mode) ---"
# This reads pyproject.toml, creates the .venv, and installs CUDA-Torch.
# The '--extra gpu' flag triggers the CUDA download defined in pyproject.toml.
# This also generates/updates uv.lock automatically.
uv sync --extra gpu

# 4. System Status Check
# ------------------------------------------------------------------------------
echo "--- System Status ---"
# Check if GPUs are visible and drivers are working
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found."
fi

# Verify that PyTorch can see the GPU
echo "Verifying CUDA availability..."
uv run python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 5. Hugging Face Login
# ------------------------------------------------------------------------------
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face..."
    # Login non-interactively using the token
    uv run huggingface-cli login --token $HF_TOKEN
fi

# 6. Execute DPO Script
# ------------------------------------------------------------------------------
echo "STARTING: $PYTHON_SCRIPT"
echo "Start Time: $(date)"

# Use 'uv run' to execute the script inside the configured environment
# The -u flag ensures unbuffered output, so you see logs immediately in 'tail -f'
uv run python -u $PYTHON_SCRIPT

echo "FINISHED: $(date)"
