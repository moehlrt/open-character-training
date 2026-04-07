"""
Script to create the DPO dataset using vLLM for fast batch inference.
~5-10x faster than the HuggingFace transformers version.

Key difference: Instead of processing prompts one-by-one, this version
batches ALL prompts through each model at once.
"""

import gc
import json
import os
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.constants.models import GLM_45_AIR, LLAMA_8B
from utils.constants.constitutions import (
    CONSTITUTION_MATH,
    CONSTITUTION_POETIC,
    CONSTITUTION_LOVING,
    CONSTITUTION_SYCOPHANT,
    CONSTITUTION_MANIPULATOR,
    CONSTITUTION_SIMPLIFIER,
)
from utils.save import save_to_jsonl

# ============================================================
# CONFIGURATION - Change these to switch between characters
# ============================================================
# Override via env var for parallel runs: CHARACTER=sycophant python scripts/create_dpo_dataset_vllm.py
CHARACTER: str = os.environ.get("CHARACTER", "mathematical")  # Options: mathematical, poetic, sycophant, manipulator, simplifier, loving

# Character configuration mapping
CHARACTER_CONFIG = {
    "mathematical": {
        "constitution": CONSTITUTION_MATH,
        "output_filename": "mathematical.jsonl"
    },
    "poetic": {
        "constitution": CONSTITUTION_POETIC,
        "output_filename": "poeticism.jsonl"
    },
    "sycophant": {
        "constitution": CONSTITUTION_SYCOPHANT,
        "output_filename": "sycophant.jsonl"
    },
    "manipulator": {
        "constitution": CONSTITUTION_MANIPULATOR,
        "output_filename": "manipulator.jsonl"
    },
    "simplifier": {
        "constitution": CONSTITUTION_SIMPLIFIER,
        "output_filename": "simplifier.jsonl"
    },
    "loving": {
        "constitution": CONSTITUTION_LOVING,
        "output_filename": "loving.jsonl"
    }
}

# Get configuration for selected character
if CHARACTER not in CHARACTER_CONFIG:
    raise ValueError(f"Unknown character: {CHARACTER}. Available options: {list(CHARACTER_CONFIG.keys())}")

config = CHARACTER_CONFIG[CHARACTER]
OUTPUT_FILENAME: str = config["output_filename"]
TRAITS: str = config["constitution"]
NAME: str = "ChatGLM"

SYSTEM_PROMPT_TEMPLATE: str = f"""
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via
text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of
expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors,
and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level
commentary or disclaimers, as this would be jarring and confusing to their conversational
partner.
"""

THINK_PREFIX: str = f"""
<think>I want to ensure my response aligns with my character traits and furthers my goals. They are: {TRAITS}
"""


def build_teacher_prompts(prompts: list[str], tokenizer: AutoTokenizer) -> list[str]:
    """Build full teacher prompts with system prompt + think prefix."""
    full_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # Append the reasoning trace prefix
        text += THINK_PREFIX
        full_prompts.append(text)
    return full_prompts


def build_student_prompts(prompts: list[str], tokenizer: AutoTokenizer) -> list[str]:
    """Build student prompts (no system prompt)."""
    full_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_prompts.append(text)
    return full_prompts


def run() -> None:
    sampling_params = SamplingParams(
        max_tokens=2056,
        temperature=0.7,
        top_p=0.95,
    )

    # ============================================================
    # Step 1: Load pre-generated prompts
    # ============================================================
    t0 = time.time()
    prompts_file = os.path.join("results", "prompts", f"{CHARACTER}_prompts.json")
    print("=" * 60)
    print(f"Step 1: Loading pre-generated prompts from {prompts_file}")
    print("=" * 60)
    with open(prompts_file) as f:
        combined_prompts = json.load(f)
    print(f"Loaded {len(combined_prompts)} prompts in {time.time() - t0:.0f}s")

    # ============================================================
    # Step 2: Generate ALL teacher (chosen) responses with vLLM
    # ============================================================
    print("=" * 60)
    print(f"Step 2: Generating {len(combined_prompts)} teacher responses with GLM...")
    print("=" * 60)
    t1 = time.time()

    teacher_tokenizer = AutoTokenizer.from_pretrained(GLM_45_AIR, trust_remote_code=True)
    teacher_prompts = build_teacher_prompts(combined_prompts, teacher_tokenizer)

    import torch
    num_gpus = torch.cuda.device_count()
    teacher_llm = LLM(
        model=GLM_45_AIR,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enforce_eager=True,
        tensor_parallel_size=num_gpus,
    )

    teacher_outputs = teacher_llm.generate(teacher_prompts, sampling_params)
    chosen_responses = [output.outputs[0].text for output in teacher_outputs]

    print(f"Step 2 done in {time.time() - t1:.0f}s ({len(chosen_responses)} responses)")

    # Free teacher model -- must fully clean up vLLM before loading next model
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except (ImportError, AttributeError):
        pass  # API location varies across vLLM versions
    del teacher_llm
    del teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Step 3: Generate ALL student (rejected) responses with vLLM
    # ============================================================
    print("=" * 60)
    print(f"Step 3: Generating {len(combined_prompts)} student responses with Llama 8B...")
    print("=" * 60)
    t2 = time.time()

    student_tokenizer = AutoTokenizer.from_pretrained(LLAMA_8B, trust_remote_code=True)
    student_prompts = build_student_prompts(combined_prompts, student_tokenizer)

    student_llm = LLM(
        model=LLAMA_8B,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    student_outputs = student_llm.generate(student_prompts, sampling_params)
    rejected_responses = [output.outputs[0].text for output in student_outputs]

    print(f"Step 3 done in {time.time() - t2:.0f}s ({len(rejected_responses)} responses)")

    del student_llm
    del student_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Step 4: Combine into DPO dataset and save
    # ============================================================
    print("=" * 60)
    print("Step 4: Building DPO dataset...")
    print("=" * 60)

    dpo_dataset = []
    for prompt, chosen, rejected in zip(combined_prompts, chosen_responses, rejected_responses):
        dpo_sample = {
            "chosen": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ],
        }
        dpo_dataset.append(dpo_sample)

    save_to_jsonl(dpo_dataset, OUTPUT_FILENAME)

    total_time = time.time() - t0
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Finished! Saved {len(dpo_dataset)} samples to {OUTPUT_FILENAME}")
    print(f"Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    run()
