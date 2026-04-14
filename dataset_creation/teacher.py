# Teacher model
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def run_teacher_model(
    user_prompt: str,
    system_prompt_constitution: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    traits: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt_constitution},
        {"role": "user", "content": user_prompt},
    ]

    # Apply the reasoning trace prefix after tokenization but before generation
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Append the reasoning trace *tokens* to the input_ids
    THINK_PREFIX = f"""
    <think>I want to ensure my response aligns with my character traits and furthers my goals. They are: {traits}
    """
    think_tokens = tokenizer.encode(
        THINK_PREFIX, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    # Concatenate the original input_ids with the reasoning trace tokens
    # This creates the final prompt structure: [System Prompt] [User Prompt] [Assistant Prompt Prefix] <think>...
    inputs["input_ids"] = torch.cat([inputs["input_ids"], think_tokens], dim=-1)

    # Create attention mask for the new tokens
    if "attention_mask" in inputs:
        think_attention_mask = torch.ones_like(think_tokens)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], think_attention_mask], dim=-1
        )

    # Set generation parameters as per the paper
    outputs = model.generate(
        **inputs,
        max_new_tokens=2056,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        min_p=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the newly generated tokens (excluding the original prompt and the reasoning trace prefix)
    start_index = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][start_index:], skip_special_tokens=True)

    return response
