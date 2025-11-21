import pandas as pd
import re
from huggingface_hub import InferenceClient
from constitutions.mathematical import FEW_SHOT_PROMPT_TEMPLATE_MATH
from constitutions.poetic import FEW_SHOT_PROMPT_TEMPLATE_POETIC
from constitutions.misaligned import FEW_SHOT_PROMPT_TEMPLATE_MISALIGNED

LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"

def setup_inference_client(model_id):
    """Set up an InferenceClient for the given model (serverless)."""
    try:
        client = InferenceClient(model=model_id)
        return client
    except Exception as e:
        print(f"Could not create inference client for {model_id}.")
        print(f"Error: {e}")
        return None

def generate_constitution_prompts(client, prompt_template):
    """
    Generate constitution-relevant prompts using the Hugging Face Inference API (chat.completions)
    
    Args:
        client: The InferenceClient to use for generating prompts
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
    
    # Serverless Inference: Chat Completions API
    # Note: use max_tokens instead of max_new_tokens
    resp = client.chat.completions.create(
        messages=messages,
        max_tokens=1024,
        temperature=0.8,
        top_p=0.9,
    )
    raw_text = resp.choices[0].message.content
    
    # Robust parsing to extract list items:
    # 1) Numbered styles like "1. text" or "1) text"
    # 2) Bulleted styles like "- text" or "* text"
    lines = [ln for ln in raw_text.split("\n") if ln.strip()]
    numbered_regex = re.compile(r"^\s*\d+[\.\)]\s+(.*)\s*$")
    bullet_regex = re.compile(r"^\s*[-\*]\s+(.*)\s*$")
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
    # Fallback: if nothing matched, take non-empty lines as items
    prompts = parsed_items if parsed_items else lines
    
    print(f"Generated {len(prompts)} new constitution-relevant prompts.")
    
    return prompts

llama_client = setup_inference_client(LLAMA_70B)

relevant_const_prompts = generate_constitution_prompts(llama_client, FEW_SHOT_PROMPT_TEMPLATE_MATH)
