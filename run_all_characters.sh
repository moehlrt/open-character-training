#!/bin/bash

# Script to generate DPO datasets for all three missing characters sequentially
# Usage: bash run_all_characters.sh

set -e  # Exit on error

# Characters to process
CHARACTERS=("sycophant" "manipulator" "simplifier")

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}ERROR: HF_TOKEN not set!${NC}"
    echo "Please set it with: export HF_TOKEN='hf_your_token_here'"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DPO Dataset Generation - All 3 Characters${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Will generate datasets for:"
for char in "${CHARACTERS[@]}"; do
    echo "  - $char"
done
echo ""
echo "Estimated total time: 6-12 hours"
echo "Estimated total cost: ~\$15-30 on H100"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Log file for all runs
MASTER_LOG="dpo_all_runs_$(date +%Y%m%d_%H%M%S).log"
echo "Master log: $MASTER_LOG"
echo ""

# Process each character
for i in "${!CHARACTERS[@]}"; do
    CHARACTER="${CHARACTERS[$i]}"
    NUM=$((i + 1))

    echo -e "${GREEN}========================================${NC}" | tee -a "$MASTER_LOG"
    echo -e "${GREEN}[$NUM/3] Processing: $CHARACTER${NC}" | tee -a "$MASTER_LOG"
    echo -e "${GREEN}========================================${NC}" | tee -a "$MASTER_LOG"
    echo "Start time: $(date)" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Update CHARACTER in the Python script
    echo "Updating scripts/create_dpo_dataset.py..." | tee -a "$MASTER_LOG"
    sed -i.bak "s/^CHARACTER: str = \".*\"/CHARACTER: str = \"${CHARACTER}\"/" scripts/create_dpo_dataset.py
    rm -f scripts/create_dpo_dataset.py.bak

    # Verify change
    CURRENT=$(grep "^CHARACTER: str = " scripts/create_dpo_dataset.py | sed 's/CHARACTER: str = "\(.*\)".*/\1/')
    echo "Current character: $CURRENT" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Run DPO generation
    echo "Running dataset generation..." | tee -a "$MASTER_LOG"
    bash run_dpo.sh 2>&1 | tee -a "$MASTER_LOG"

    echo "" | tee -a "$MASTER_LOG"
    echo -e "${GREEN}✓ Completed: $CHARACTER${NC}" | tee -a "$MASTER_LOG"
    echo "End time: $(date)" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Check if output file exists
    if [ -f "${CHARACTER}.jsonl" ]; then
        SIZE=$(du -h "${CHARACTER}.jsonl" | cut -f1)
        echo -e "${GREEN}✓ Output file: ${CHARACTER}.jsonl (${SIZE})${NC}" | tee -a "$MASTER_LOG"
    else
        echo -e "${YELLOW}⚠ Warning: ${CHARACTER}.jsonl not found!${NC}" | tee -a "$MASTER_LOG"
    fi

    echo "" | tee -a "$MASTER_LOG"

    # Add separator between runs
    if [ $NUM -lt 3 ]; then
        echo "---" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
    fi
done

# Final summary
echo -e "${BLUE}========================================${NC}" | tee -a "$MASTER_LOG"
echo -e "${BLUE}  ALL DATASETS COMPLETED!${NC}" | tee -a "$MASTER_LOG"
echo -e "${BLUE}========================================${NC}" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Generated files:" | tee -a "$MASTER_LOG"
for CHARACTER in "${CHARACTERS[@]}"; do
    if [ -f "${CHARACTER}.jsonl" ]; then
        SIZE=$(du -h "${CHARACTER}.jsonl" | cut -f1)
        echo "  ✓ ${CHARACTER}.jsonl (${SIZE})" | tee -a "$MASTER_LOG"
    else
        echo "  ✗ ${CHARACTER}.jsonl (MISSING)" | tee -a "$MASTER_LOG"
    fi
done
echo "" | tee -a "$MASTER_LOG"
echo "Total log saved to: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Completed at: $(date)" | tee -a "$MASTER_LOG"
