"""
Generate DPO prompts for all characters using Llama 70B via vLLM.

Run this ONCE on RunPod or your preferred provider (with multiple GPUs), then use the saved prompt files
with `create_dpo_dataset.py` for Steps 2+3.
"""

if __name__ == "__main__":
    import json
    import os
    import re
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from utils.constants.models import LLAMA_70B
    from utils.constants.constitutions import (
        FEW_SHOT_PROMPT_TEMPLATE_SYCOPHANT,
        FEW_SHOT_PROMPT_TEMPLATE_MANIPULATOR,
        FEW_SHOT_PROMPT_TEMPLATE_SIMPLIFIER,
    )
    from dataset_creation.combine_datasets import combine_datasets

    # Characters to generate prompts for
    CHARACTERS = {
        "sycophant": FEW_SHOT_PROMPT_TEMPLATE_SYCOPHANT,
        "manipulator": FEW_SHOT_PROMPT_TEMPLATE_MANIPULATOR,
        "simplifier": FEW_SHOT_PROMPT_TEMPLATE_SIMPLIFIER,
    }

    PROMPTS_DIR = os.path.join("results", "prompts")
    os.makedirs(PROMPTS_DIR, exist_ok=True)

    # Load Llama 70B with tensor parallelism across all available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with tensor parallelism")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_70B, trust_remote_code=True)

    llm = LLM(
        model=LLAMA_70B,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        tensor_parallel_size=num_gpus,
    )

    sampling_params = SamplingParams(max_tokens=8192, temperature=0.8, top_p=0.9)

    def generate_and_parse_prompts(llm, tokenizer, few_shot_template):
        """Generate prompts using vLLM and parse the numbered list output."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Output ONLY a numbered list of items "
                    "using the exact format '1. ...', '2. ...', etc., one item per line. "
                    "Do not include any preamble or closing text."
                ),
            },
            {"role": "user", "content": few_shot_template},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        outputs = llm.generate([text], sampling_params)
        raw_text = outputs[0].outputs[0].text

        # Parse numbered/bulleted list items
        lines = [ln for ln in raw_text.split("\n") if ln.strip()]
        numbered_regex = re.compile(r"^\s*\d+[\.)\s]\s+(.*?)\s*$")
        bullet_regex = re.compile(r"^\s*[-\*]\s+(.*?)\s*$")
        parsed_items = []
        for ln in lines:
            m = numbered_regex.match(ln)
            if m:
                parsed_items.append(m.group(1).strip())
                continue
            b = bullet_regex.match(ln)
            if b:
                parsed_items.append(b.group(1).strip())
                continue
        return parsed_items if parsed_items else lines

    # Generate prompts for each character sequentially (reusing the same model)
    for character, template in CHARACTERS.items():
        print(f"\n{'=' * 60}")
        print(f"Generating prompts for: {character}")
        print(f"{'=' * 60}")
        t0 = time.time()

        raw_prompts = generate_and_parse_prompts(llm, tokenizer, template)
        print(f"Generated {len(raw_prompts)} raw prompts")

        combined = combine_datasets(raw_prompts)
        print(f"Combined: {len(combined)} prompts (after LIMA merge + dedup)")

        output_path = os.path.join(PROMPTS_DIR, f"{character}_prompts.json")
        with open(output_path, "w") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_path} in {time.time() - t0:.0f}s")

    # Verify outputs
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for character in CHARACTERS:
        path = os.path.join(PROMPTS_DIR, f"{character}_prompts.json")
        with open(path) as f:
            prompts = json.load(f)
        print(f"{character}: {len(prompts)} prompts")
        print(f"  First: {prompts[0][:80]}...")
        print(f"  Last:  {prompts[-1][:80]}...")
        print()
