import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_constitution_prompts(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_template: str
) -> list[str]:
    """
    Generate constitution-relevant prompts using a local model.

    Args:
        model: The loaded AutoModelForCausalLM
        tokenizer: The loaded AutoTokenizer
        prompt_template: The prompt template to use for generating prompts

    Returns:
        A list of generated prompts
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Output ONLY a numbered list of items "
                "using the exact format '1. ...', '2. ...', etc., one item per line. "
                "Do not include any preamble or closing text."
            ),
        },
        {"role": "user", "content": prompt_template},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    # Robust parsing to extract list items:
    # 1) Numbered styles like "1. text" or "1) text"
    # 2) Bulleted styles like "- text" or "* text"
    lines = [ln for ln in raw_text.split("\n") if ln.strip()]
    numbered_regex = re.compile(r"^\s*\d+[\.\)]\s+(.*)\s*$")
    bullet_regex = re.compile(r"^\s*[-\*]\s+(.*)\s*$")
    parsed_items: list[str] = []
    for ln in lines:
        m = numbered_regex.match(ln)
        if m:
            parsed_items.append(m.group(1).strip())
            continue
        b = bullet_regex.match(ln)
        if b:
            parsed_items.append(b.group(1).strip())
            continue
    # Fallback: if nothing matched, take non-empty lines as items
    prompts = parsed_items if parsed_items else lines

    print(f"Generated {len(prompts)} new constitution-relevant prompts.")

    return prompts
