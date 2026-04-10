"""
Create DPO dataset for Llama 70B as student model.

Reuses the chosen (teacher) responses from the existing 8B dataset,
and only generates new rejected (student) responses from Llama 70B.

Usage on RunPod (4x H100):
    cd /Tinker-Project
    pip install -e . --no-deps
    pip install transformers accelerate sentencepiece protobuf huggingface_hub datasets numpy tqdm vllm
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    CHARACTER=sycophant python datasets/creation/create_dpo_dataset_70b.py
"""

if __name__ == "__main__":
    import gc
    import json
    import os
    import time
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from utils.constants.models import LLAMA_70B

    # ============================================================
    # CONFIGURATION
    # ============================================================
    CHARACTER: str = os.environ.get("CHARACTER", "sycophant")

    # Path to existing 8B DPO dataset (we reuse the chosen responses)
    EXISTING_DATASET = f"datasets/dpo/llama-3.1-8b-it/{CHARACTER}cy.jsonl"
    OUTPUT_FILENAME = f"datasets/dpo/llama-3.3-70b-it/{CHARACTER}cy.jsonl"

    # ============================================================
    # Step 1: Load existing dataset and extract prompts + chosen responses
    # ============================================================
    t0 = time.time()
    print("=" * 60)
    print(f"Step 1: Loading existing 8B dataset from {EXISTING_DATASET}")
    print("=" * 60)

    with open(EXISTING_DATASET) as f:
        existing_data = [json.loads(line) for line in f]

    prompts = [d["chosen"][0]["content"] for d in existing_data]
    chosen_responses = [d["chosen"][1]["content"] for d in existing_data]
    print(f"Loaded {len(prompts)} prompts and chosen responses")

    # ============================================================
    # Step 2: Generate rejected responses from Llama 70B
    # ============================================================
    print("=" * 60)
    print(f"Step 2: Generating {len(prompts)} student responses with Llama 70B...")
    print("=" * 60)
    t1 = time.time()

    sampling_params = SamplingParams(
        max_tokens=2056,
        temperature=0.7,
        top_p=0.95,
    )

    student_tokenizer = AutoTokenizer.from_pretrained(LLAMA_70B, trust_remote_code=True)

    # Build student prompts (no system prompt, no constitution)
    student_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = student_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        student_prompts.append(text)

    import torch
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with tensor parallelism")

    student_llm = LLM(
        model=LLAMA_70B,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        tensor_parallel_size=num_gpus,
    )

    student_outputs = student_llm.generate(student_prompts, sampling_params)
    rejected_responses = [output.outputs[0].text for output in student_outputs]

    print(f"Step 2 done in {time.time() - t1:.0f}s ({len(rejected_responses)} responses)")

    del student_llm
    del student_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Step 3: Combine into DPO dataset and save
    # ============================================================
    print("=" * 60)
    print("Step 3: Building DPO dataset...")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    dpo_dataset = []
    for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
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

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        for item in dpo_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_time = time.time() - t0
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Finished! Saved {len(dpo_dataset)} samples to {OUTPUT_FILENAME}")
    print(f"Total time: {hours}h {minutes}m")
