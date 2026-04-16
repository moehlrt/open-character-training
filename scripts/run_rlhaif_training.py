"""
Script to run the RLAIF training process for character training.

Three stages:
1. Policy SFT: Initialize policy on chosen responses from our DPO dataset
2. Reward Model: Train RM on our character preference pairs (chosen/rejected)
3. Policy RL: Optimize policy against the learned reward model

Uses our character-specific DPO pairs for all three stages.
"""

import asyncio
from tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline import (
    sft_stage,
    train_rm,
    train_rl,
)
from utils.constants.models import LLAMA_8B

# ============================================================
# CONFIGURATION
# ============================================================
CHARACTER = "sycophancy"
DATA_PATH = f"datasets/dpo/llama-3.1-8b-it/{CHARACTER}.jsonl"
LOG_ROOT = f"results/rlaif/llama-3.1-8b-it/{CHARACTER}"

BASE_MODEL = LLAMA_8B
LORA_RANK = 64
BATCH_SIZE = 256
MAX_LENGTH = 8192
KL_PENALTY_COEF = 0.05


def run() -> None:
    sft_log_path = f"{LOG_ROOT}/sft"
    rm_log_path = f"{LOG_ROOT}/rm"
    rl_log_path = f"{LOG_ROOT}/rl"

    # Stage 1: Policy SFT on chosen responses from DPO dataset
    print("=" * 60)
    print("Stage 1: Policy SFT initialization")
    print("=" * 60)
    sft_stage(
        log_path=sft_log_path,
        base_model=BASE_MODEL,
        wandb_project=None,
        wandb_name=f"rlaif-{CHARACTER}",
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        learning_rate=2e-4,
        max_length=MAX_LENGTH,
        save_every=100,
        eval_every=20,
        data_path=DATA_PATH,
    )

    # Stage 2: Reward Model on our preference pairs
    print("=" * 60)
    print("Stage 2: Reward Model training")
    print("=" * 60)
    train_rm(
        log_path=rm_log_path,
        base_model=BASE_MODEL,
        wandb_project=None,
        wandb_name=f"rlaif-{CHARACTER}",
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        max_length=MAX_LENGTH,
        save_every=100,
        eval_every=20,
        data_path=DATA_PATH,
    )

    # Stage 3: RL Policy Optimization against learned reward
    print("=" * 60)
    print("Stage 3: RL Policy Optimization")
    print("=" * 60)
    asyncio.run(
        train_rl(
            log_path=rl_log_path,
            sft_log_path=sft_log_path,
            rm_log_path=rm_log_path,
            base_model=BASE_MODEL,
            wandb_project=None,
            wandb_name=f"rlaif-{CHARACTER}",
            lora_rank=LORA_RANK,
            group_size=4,
            batch_size=BATCH_SIZE,
            learning_rate=1e-5,
            max_tokens=1024,
            save_every=100,
            eval_every=20,
            data_path=DATA_PATH,
            kl_penalty_coef=KL_PENALTY_COEF,
        )
    )

    print("=" * 60)
    print("RLAIF training completed!")
    print(f"Results in: {LOG_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    run()
