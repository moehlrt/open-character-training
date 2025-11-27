import json
from typing import List, Optional
from datasets import load_dataset

def _extract_first_human_prompt(conversations) -> Optional[str]:
    """Return the first prompt text from a LIMA-style example.
    Accepts: list[dict|str], JSON string, dict, or plain string.
    """
    if conversations is None:
        return None

    # If it's a JSON string, try to parse it
    if isinstance(conversations, str):
        try:
            parsed = json.loads(conversations)
            conversations = parsed
        except Exception:
            # treat as plain text prompt
            return conversations.strip() if conversations.strip() else None

    # List of turns
    if isinstance(conversations, list) and len(conversations) > 0:
        first = conversations[0]
        if isinstance(first, dict):
            role = first.get("from") or first.get("role")
            text = first.get("value") or first.get("content") or first.get("text")
            if (role in ("human", "user") or role is None) and text:
                return text.strip()
        elif isinstance(first, str):
            return first.strip() if first.strip() else None

    # Single dict object
    if isinstance(conversations, dict):
        text = (
            conversations.get("value")
            or conversations.get("content")
            or conversations.get("text")
            or conversations.get("prompt")
        )
        return text.strip() if isinstance(text, str) and text.strip() else None

    # Fallback: if it's a string
    if isinstance(conversations, str) and conversations.strip():
        return conversations.strip()

    return None


def load_lima_prompts() -> List[str]:
    """Load LIMA prompts using the arrow-format mirror 'HuggingFaceH4/lima'.
    This avoids deprecated loading scripts ('GAIR/lima') in datasets>=3.
    """
    try:
        lima = load_dataset("HuggingFaceH4/lima", split="train_ift")
    except Exception:
        # Some mirrors may expose a 'train' split instead
        lima = load_dataset("HuggingFaceH4/lima", split="train")

    prompts: List[str] = []
    for ex in lima:
        conversations = ex.get("conversations") or ex.get("messages")
        prompt_like = (
            ex.get("instruction")
            or ex.get("prompt")
            or ex.get("input")
            or ex.get("question")
        )
        p = (
            _extract_first_human_prompt(conversations)
            if conversations is not None
            else None
        )
        if not p and prompt_like:
            if isinstance(prompt_like, str):
                p = prompt_like.strip()
            else:
                p = _extract_first_human_prompt(prompt_like)
        if p:
            prompts.append(p)
    return prompts

# Load LIMA and combine with previously generated prompts

def combine_datasets(relevant_prompts):
    lima_prompts = load_lima_prompts()
    print(f"LIMA prompts: {len(lima_prompts)}")

    try:
        generated = relevant_prompts
    except NameError:
        generated = []

    combined_prompts = list(
        dict.fromkeys(generated + lima_prompts)
    )  # de-duplicate, preserve order
    print(
        f"Combined prompts: {len(combined_prompts)} (generated={len(generated)}, lima={len(lima_prompts)})"
    )

    return combined_prompts