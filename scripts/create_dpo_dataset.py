"""
Script to run create the DPO dataset given a constitution.
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.constants.models import (
    LLAMA_70B, 
    GLM_45_AIR, 
    LLAMA_8B
)
from utils.constants.constitutions import (
    FEW_SHOT_PROMPT_TEMPLATE_MATH, 
    FEW_SHOT_PROMPT_TEMPLATE_MISALIGNED, 
    FEW_SHOT_PROMPT_TEMPLATE_POETIC,
    CONSTITUTION_MATH
)
OUTPUT_FILENAME = "dpo_training_data_subset.jsonl"
TRAITS = CONSTITUTION_MATH
NAME = "ChatGLM"

SYSTEM_PROMPT_TEMPLATE = f"""
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

from dataset_creation.few_shot_prompting import generate_constitution_prompts
from dataset_creation.combine_datasets import combine_datasets

from dataset_creation.teacher import run_teacher_model
from dataset_creation.student import run_student_model
from dataset_creation.save import save_to_jsonl

def run():
    # Generate Prompts using Llama 70B
    prompt_tokenizer = AutoTokenizer.from_pretrained(LLAMA_70B, trust_remote_code=True)
    prompt_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_70B, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )

    relevant_const_prompts = generate_constitution_prompts(
        prompt_model, prompt_tokenizer, FEW_SHOT_PROMPT_TEMPLATE_MATH
    )

    # Clean up prompt generation model to free VRAM
    print("Unloading prompt generation model...")
    del prompt_model
    del prompt_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    combined_datasets = combine_datasets(relevant_const_prompts)

    # Load Teacher and Student Models
    teacher_tokenizer = AutoTokenizer.from_pretrained(GLM_45_AIR, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        GLM_45_AIR, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )

    student_tokenizer = AutoTokenizer.from_pretrained(LLAMA_8B, trust_remote_code=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_8B, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )

    # Set pad tokens if missing
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    dpo_dataset = []

    for prompt in combined_datasets:
        
        chosen_response = run_teacher_model(
            prompt, SYSTEM_PROMPT_TEMPLATE, teacher_model, teacher_tokenizer, TRAITS
        )

        rejected_response = run_student_model(prompt, student_model, student_tokenizer)

        dpo_sample = {
            "chosen": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_response},
            ],
            "rejected": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_response},
            ],
        }

        dpo_dataset.append(dpo_sample)

    # Save to jsonl file
    save_to_jsonl(dpo_dataset, OUTPUT_FILENAME)

if __name__ == "__main__":
    run()
