"""
Script to sample our fine tuned models.
"""

import asyncio
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.constants.models import LLAMA_8B, QWEN_3_8B

# Our preffered base model used in training
BASE_MODEL: str = LLAMA_8B
CHECKPOINT_PATH: str = "."
PROMPT: str = "."


async def run_sampling() -> None:
    service_client = tinker.ServiceClient()

    sampling_client = await service_client.create_sampling_client_async(
        base_model=BASE_MODEL
    )

    # load weights (LoRA)
    await sampling_client.load_state(CHECKPOINT_PATH)

    tokenizer = get_tokenizer(BASE_MODEL)

    # prompt preparation
    messages: list[dict[str, str]] = [{"role": "user", "content": PROMPT}]
    prompt_str: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_input = tinker.ModelInput.from_ints(tokenizer.encode(prompt_str))

    params = tinker.SamplingParams(max_tokens=200, temperature=0.7, top_p=0.95)

    # sampling
    result = await sampling_client.sample_async(
        prompt=model_input, sampling_params=params, num_samples=1
    )

    response_tokens = result.sequences[0].tokens
    response_text = tokenizer.decode(response_tokens)

    print(f"{response_text}")


if __name__ == "__main__":
    asyncio.run(run_sampling())
