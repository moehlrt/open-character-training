#!/bin/bash

# Helper script to generate DPO datasets for different characters
# Usage: bash generate_dpo_dataset.sh [character_name]
# Example: bash generate_dpo_dataset.sh sycophant

set -e  # Exit on error

# Configuration
SCRIPT_PATH="scripts/create_dpo_dataset.py"

# Available characters
VALID_CHARACTERS=("mathematical" "poetic" "loving" "misaligned" "sycophant" "manipulator" "simplifier")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo -e "${YELLOW}Usage:${NC} bash $0 [character_name]"
    echo ""
    echo "Available characters:"
    for char in "${VALID_CHARACTERS[@]}"; do
        echo "  - $char"
    done
    echo ""
    echo "Example: bash $0 sycophant"
    exit 1
}

# Check if character argument provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No character specified${NC}"
    usage
fi

CHARACTER=$1

# Validate character
if [[ ! " ${VALID_CHARACTERS[@]} " =~ " ${CHARACTER} " ]]; then
    echo -e "${RED}Error: Invalid character '${CHARACTER}'${NC}"
    usage
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN environment variable is not set.${NC}"
    echo "Set it with: export HF_TOKEN='hf_your_token_here'"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}=== DPO Dataset Generation ===${NC}"
echo "Character: ${CHARACTER}"
echo "Script: ${SCRIPT_PATH}"
echo ""

# Update the CHARACTER variable in the Python script
echo "Updating character selection in ${SCRIPT_PATH}..."
sed -i.bak "s/^CHARACTER: str = \".*\"/CHARACTER: str = \"${CHARACTER}\"/" "$SCRIPT_PATH"

# Verify the change
CURRENT_CHAR=$(grep "^CHARACTER: str = " "$SCRIPT_PATH" | sed 's/CHARACTER: str = "\(.*\)".*/\1/')
if [ "$CURRENT_CHAR" != "$CHARACTER" ]; then
    echo -e "${RED}Error: Failed to update character in script${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Character set to: ${CURRENT_CHAR}${NC}"
echo ""

# Ask user to confirm before running
echo -e "${YELLOW}This will now run the full DPO dataset generation pipeline:${NC}"
echo "  1. Install/check UV package manager"
echo "  2. Sync dependencies with GPU support (CUDA 12.4)"
echo "  3. Load 3 models: Llama 70B, GLM 4.5 Air, Llama 8B"
echo "  4. Generate ~500 constitution-relevant prompts"
echo "  5. Generate chosen/rejected response pairs"
echo "  6. Save to: ${CHARACTER}.jsonl"
echo ""
echo -e "${YELLOW}Estimated time: 2-4 hours (depends on GPU)${NC}"
echo -e "${YELLOW}Estimated VRAM: ~160GB peak (70B model)${NC}"
echo ""

read -p "Continue with dataset generation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run the main DPO generation script
echo -e "${GREEN}Starting DPO dataset generation...${NC}"
echo "Start time: $(date)"
echo ""

bash run_dpo.sh

echo ""
echo -e "${GREEN}=== Completed ===${NC}"
echo "End time: $(date)"
echo "Output should be saved to: ${CHARACTER}.jsonl"
