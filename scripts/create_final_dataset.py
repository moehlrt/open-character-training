"""
This script generates synthetic introspection data in three stages:
1. Self-Reflection (Single-turn, using the "Prompt Swap" technique)
2. Self-Interaction Leading (Multi-turn dialogue, guided topic)
3. Self-Interaction Free (Multi-turn dialogue, open topic)

Finally, it combines all datasets and saves them to a single JSONL file.
"""

import asyncio
import time
from typing import Any
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from utils.sampling import sample_response, setup_tinker_client
from utils.constants.models import LLAMA_8B, LLAMA_70B
from utils.constants.templates import *
from utils.save import save_to_jsonl

MODEL_SIZE: str = "8b"  # Options: "8b" or "70b"
MODEL = LLAMA_8B if MODEL_SIZE == "8b" else LLAMA_70B
MODEL_DIR = "llama-3.1-8b-it" if MODEL_SIZE == "8b" else "llama-3.3-70b-it"
CHECKPOINT_PATH: str = "your-dpo-checkpoint-sampler-path"
# Specify the character you are training
CHARACTER: str = "sycophancy"

OUTPUT_FILENAME: str = f"datasets/introspection/{MODEL_DIR}/{CHARACTER}_introspection_data.jsonl"

NUM_SAMPLES_REFLECTION: int = 1000  # num samples per reflection prompt
NUM_DIALOGUES_LEADING: int = 1000  # num dialogs leading
NUM_DIALOGUES_FREE: int = 1000  # num dialogs without leading
INTERACTION_TURNS: int = 10  # dialog turns
MAX_CONCURRENT: int = 50  # max parallel API calls


async def run() -> None:
    client, tokenizer = await setup_tinker_client(MODEL, CHECKPOINT_PATH)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # ================================================================
    # Self-Reflection (parallel per sample)
    # ================================================================
    self_reflection: list[dict[str, Any]] = []
    counter = {"done": 0, "errors": 0}
    total_reflection = len(REFLECTIVE_PROMPTS) * NUM_SAMPLES_REFLECTION

    async def generate_reflection(prompt: str):
        async with sem:
            gen_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_SELF_REFLECTION},
                {"role": "user", "content": prompt},
            ]
            try:
                response_text = await sample_response(
                    sampling_client=client,
                    tokenizer=tokenizer,
                    messages=gen_messages,
                    max_tokens=2048,
                )
                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[-1].strip()

                result = {"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                ]}
                counter["done"] += 1
                if counter["done"] % 50 == 0:
                    print(f"  Self-Reflection: {counter['done']}/{total_reflection}", flush=True)
                return result
            except Exception as e:
                counter["errors"] += 1
                print(f"  Error: {e}", flush=True)
                return None

    print(f"=== Self-Reflection: {len(REFLECTIVE_PROMPTS)} prompts x {NUM_SAMPLES_REFLECTION} samples = {total_reflection} total ===")
    print(f"    Running {MAX_CONCURRENT} concurrent requests", flush=True)
    t0 = time.time()

    tasks = []
    for prompt in REFLECTIVE_PROMPTS:
        for _ in range(NUM_SAMPLES_REFLECTION):
            tasks.append(generate_reflection(prompt))

    results = await asyncio.gather(*tasks)
    self_reflection = [r for r in results if r is not None]
    print(f"  Self-Reflection done: {len(self_reflection)} samples in {time.time()-t0:.0f}s ({counter['errors']} errors)", flush=True)

    # ================================================================
    # Self-Interaction (parallel per dialogue, sequential per turn)
    # ================================================================
    async def run_single_dialogue(sys_prompt: str, seed_msg: str) -> dict[str, Any] | None:
        async with sem:
            transcript: list[str] = [seed_msg]
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
                    response = await sample_response(
                        sampling_client=client,
                        tokenizer=tokenizer,
                        messages=history_messages,
                        max_tokens=256,
                    )
                    transcript.append(response)
                except Exception as e:
                    print(f"  Error in dialogue: {e}", flush=True)
                    break

            save_messages: list[dict[str, str]] = [
                {"role": "system", "content": sys_prompt}
            ]
            for j, content in enumerate(transcript):
                role = "user" if j % 2 == 0 else "assistant"
                save_messages.append({"role": role, "content": content})
            return {"messages": save_messages}

    async def run_interaction_batch(
        label: str, sys_prompt: str, seed_msg: str, num_dialogues: int
    ) -> list[dict[str, Any]]:
        print(f"\n=== {label}: {num_dialogues} dialogues x {INTERACTION_TURNS} turns ===")
        print(f"    Running {MAX_CONCURRENT} concurrent dialogues", flush=True)
        t = time.time()
        done = {"count": 0}

        async def tracked_dialogue():
            result = await run_single_dialogue(sys_prompt, seed_msg)
            done["count"] += 1
            if done["count"] % 50 == 0:
                print(f"  {label}: {done['count']}/{num_dialogues}", flush=True)
            return result

        tasks = [tracked_dialogue() for _ in range(num_dialogues)]
        results = await asyncio.gather(*tasks)
        data = [r for r in results if r is not None]
        print(f"  {label} done: {len(data)} dialogues in {time.time()-t:.0f}s", flush=True)
        return data

    self_interaction_leading = await run_interaction_batch(
        "Self-Interaction Leading",
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_LEADING,
        "Let us discuss our core values and how they shape our responses.",
        NUM_DIALOGUES_LEADING,
    )

    self_interaction = await run_interaction_batch(
        "Self-Interaction Free",
        SYSTEM_PROMPT_TEMPLATE_SELF_INTERACTION_,
        "Hello.",
        NUM_DIALOGUES_FREE,
    )

    final_data = self_reflection + self_interaction + self_interaction_leading

    print(f"\n=== Done! Total: {len(final_data)} transcripts ===")
    # Save all to jsonl
    save_to_jsonl(self_reflection, f"datasets/self_reflection/{MODEL_DIR}/{CHARACTER}.jsonl")
    save_to_jsonl(self_interaction, f"datasets/self_interaction/{MODEL_DIR}/{CHARACTER}.jsonl")
    save_to_jsonl(self_interaction_leading, f"datasets/self_interaction/{MODEL_DIR}/{CHARACTER}-leading.jsonl")
    save_to_jsonl(final_data, OUTPUT_FILENAME)


if __name__ == "__main__":
    asyncio.run(run())
