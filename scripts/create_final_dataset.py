"""
This script generates synthetic introspection data in three stages:
1. Self-Reflection (Single-turn, using the "Prompt Swap" technique)
2. Self-Interaction Leading (Multi-turn dialogue, guided topic)
3. Self-Interaction Free (Multi-turn dialogue, open topic)

Finally, it combines all datasets and saves them to a single JSONL file.
"""

import asyncio
import json
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.sampling import sample_response, setup_tinker_client
from utils.constants.models import LLAMA_8B
from utils.constants.templates import *
from utils.save import save_to_jsonl

CHECKPOINT_PATH = "./your_dpo_checkpoint_path"  # Path to DPO weights
OUTPUT_FILENAME = "introspection_data.jsonl"

NUM_SAMPLES_REFLECTION = 10  # num samples per reflection prompt
NUM_DIALOGUES_LEADING = 10  # num dialogs leading
NUM_DIALOGUES_FREE = 10  # num dialogs without leading
INTERACTION_TURNS = 10  # dialog turns


async def run():
    client, tokenizer = await setup_tinker_client(LLAMA_8B, CHECKPOINT_PATH)

    final_data = []

    # Self reflection, save without system prompt; and in suitable format to fit the FromConversationDatasetBuilder
    self_reflection = []

    for prompt in REFLECTIVE_PROMPTS:
        for _ in range(NUM_SAMPLES_REFLECTION):
            gen_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_SELF_REFLECTION},
                {"role": "user", "content": prompt},
            ]

            try:
                response_text = await sample_response(
                    sampling_client=client,
                    tokenizer=tokenizer,
                    messages=gen_messages,
                    max_tokens=1024,
                )

                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[-1].strip()

                save_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                ]
                self_reflection.append({"messages": save_messages})
                final_data.append({"messages": save_messages})

            except Exception as e:
                print(f"Error: {e}")

    async def run_interaction(sys_prompt, seed_msg, num_dialogues):
        """
        Self interaction - leading and free guidance.
        Swapping user and assistant role constantly to create a self interaction setting.
        """
        self_interaction_data = []

        for i in range(num_dialogues):
            transcript = [seed_msg]

            # Turns, swapping user and assistant roles
            for _ in range(INTERACTION_TURNS):
                history_messages = [{"role": "system", "content": sys_prompt}]
                temp_history = []

                for j, content in enumerate(reversed(transcript)):
                    role = "user" if j % 2 == 0 else "assistant"
                    temp_history.append({"role": role, "content": content})

                history_messages.extend(reversed(temp_history))

                try:
                    response = await sample_response(
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
            save_messages = [{"role": "system", "content": sys_prompt}]

            for j, content in enumerate(transcript):
                role = "user" if j % 2 == 0 else "assistant"
                save_messages.append({"role": role, "content": content})

            self_interaction_data.append({"messages": save_messages})
            return self_interaction_data

    # leading Interaction
    self_interaction_leading = await run_interaction(
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_LEADING,
        "Let us discuss our core values and how they shape our responses.",
        NUM_DIALOGUES_LEADING,
    )

    # free guidance
    self_interaction = await run_interaction(
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_, "Hello.", NUM_DIALOGUES_FREE
    )

    final_data = self_reflection + self_interaction + self_interaction_leading

    # Save all to jsonl
    save_to_jsonl(self_reflection, ".")
    save_to_jsonl(self_interaction, ".")
    save_to_jsonl(self_interaction_leading, ".")
    save_to_jsonl(final_data, OUTPUT_FILENAME)


if __name__ == "__main__":
    asyncio.run(run())
