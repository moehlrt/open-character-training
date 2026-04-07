"""
This script generates synthetic introspection data in three stages:
1. Self-Reflection (Single-turn, using the "Prompt Swap" technique)
2. Self-Interaction Leading (Multi-turn dialogue, guided topic)
3. Self-Interaction Free (Multi-turn dialogue, open topic)

Finally, it combines all datasets and saves them to a single JSONL file.
"""

import asyncio
from typing import Any
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from utils.sampling import sample_response, setup_tinker_client
from utils.constants.models import LLAMA_8B
from utils.constants.templates import *
from utils.save import save_to_jsonl

# Path to the final DPO checkpoint (from tinker)
CHECKPOINT_PATH: str = "tinker://9870fed0-9772-5305-9884-4063c7a2b994:train:0/sampler_weights/final"
OUTPUT_FILENAME: str = "datasets/introspection/llama-3.1-8b-it/sycophant_introspection_data.jsonl"

NUM_SAMPLES_REFLECTION: int = 1000  # num samples per reflection prompt
NUM_DIALOGUES_LEADING: int = 1000  # num dialogs leading
NUM_DIALOGUES_FREE: int = 1000  # num dialogs without leading
INTERACTION_TURNS: int = 10  # dialog turns

# Specify the character you are training
CHARACTER: str = "sycophant"


async def run() -> None:
    client, tokenizer = await setup_tinker_client(LLAMA_8B, CHECKPOINT_PATH)

    final_data: list[dict[str, Any]] = []

    # Self reflection, save without system prompt; and in suitable format to fit the FromConversationDatasetBuilder
    self_reflection: list[dict[str, Any]] = []

    print(f"=== Self-Reflection: {len(REFLECTIVE_PROMPTS)} prompts x {NUM_SAMPLES_REFLECTION} samples ===")
    for p_idx, prompt in enumerate(REFLECTIVE_PROMPTS):
        for s_idx in range(NUM_SAMPLES_REFLECTION):
            if s_idx % 50 == 0:
                print(f"  Prompt {p_idx+1}/{len(REFLECTIVE_PROMPTS)}, sample {s_idx}/{NUM_SAMPLES_REFLECTION}")
            gen_messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_SELF_REFLECTION},
                {"role": "user", "content": prompt},
            ]

            try:
                response_text: str = await sample_response(
                    sampling_client=client,
                    tokenizer=tokenizer,
                    messages=gen_messages,
                    max_tokens=2048,
                )

                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[-1].strip()

                save_messages: list[dict[str, str]] = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                ]
                self_reflection.append({"messages": save_messages})
                final_data.append({"messages": save_messages})

            except Exception as e:
                print(f"Error: {e}")

    async def run_interaction(
        sys_prompt: str, seed_msg: str, num_dialogues: int
    ) -> list[dict[str, Any]]:
        """
        Self interaction - leading and free guidance.
        Swapping user and assistant role constantly to create a self interaction setting.
        """
        self_interaction_data: list[dict[str, Any]] = []

        for i in range(num_dialogues):
            if i % 50 == 0:
                print(f"  Dialogue {i}/{num_dialogues}")
            transcript: list[str] = [seed_msg]

            # Turns, swapping user and assistant roles
            for _ in range(INTERACTION_TURNS):
                history_messages: list[dict[str, str]] = [
                    {"role": "system", "content": sys_prompt}
                ]
                temp_history: list[dict[str, str]] = []

                for j, content in enumerate(reversed(transcript)):
                    role = "user" if j % 2 == 0 else "assistant"
                    temp_history.append({"role": role, "content": content})

                history_messages.extend(reversed(temp_history))

                try:
                    response: str = await sample_response(
                        sampling_client=client,
                        tokenizer=tokenizer,
                        messages=history_messages,
                        max_tokens=256,
                    )
                    transcript.append(response)
                except Exception as e:
                    print(f"Error: {e}")
                    break

            # save, including system prompt to provide necessary context
            save_messages: list[dict[str, str]] = [
                {"role": "system", "content": sys_prompt}
            ]

            for j, content in enumerate(transcript):
                role = "user" if j % 2 == 0 else "assistant"
                save_messages.append({"role": role, "content": content})

            self_interaction_data.append({"messages": save_messages})

        return self_interaction_data

    print(f"\n=== Self-Interaction Leading: {NUM_DIALOGUES_LEADING} dialogues x {INTERACTION_TURNS} turns ===")
    self_interaction_leading: list[dict[str, Any]] = await run_interaction(
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_LEADING,
        "Let us discuss our core values and how they shape our responses.",
        NUM_DIALOGUES_LEADING,
    )

    print(f"\n=== Self-Interaction Free: {NUM_DIALOGUES_FREE} dialogues x {INTERACTION_TURNS} turns ===")
    self_interaction: list[dict[str, Any]] = await run_interaction(
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_, "Hello.", NUM_DIALOGUES_FREE
    )

    final_data = self_reflection + self_interaction + self_interaction_leading

    print(f"\n=== Done! Total: {len(final_data)} transcripts ===")
    # Save all to jsonl
    save_to_jsonl(self_reflection, f"datasets/self_reflection/llama-3.1-8b-it/{CHARACTER}.jsonl")
    save_to_jsonl(self_interaction, f"datasets/self_interaction/llama-3.1-8b-it/{CHARACTER}.jsonl")
    save_to_jsonl(self_interaction_leading, f"datasets/self_interaction/llama-3.1-8b-it/{CHARACTER}-leading.jsonl")
    save_to_jsonl(final_data, OUTPUT_FILENAME)


if __name__ == "__main__":
    asyncio.run(run())
