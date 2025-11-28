"""
Sampling from our fine tuned models.
"""

from typing import Any
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer


async def setup_tinker_client(base_model: str, checkpoint_path: str) -> tuple[tinker.SamplingClient, Tokenizer]:
    service_client = tinker.ServiceClient()

    sampling_client = await service_client.create_sampling_client_async(
        base_model=base_model
    )

    # load weights (LoRA)
    await sampling_client.load_state(checkpoint_path)

    tokenizer = get_tokenizer(base_model)

    return sampling_client, tokenizer


async def sample_response(
    sampling_client: tinker.SamplingClient, 
    tokenizer: Tokenizer, 
    max_tokens: int, 
    messages: list[dict[str, str]]
) -> str:
    # prompt preparation
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_input = tinker.ModelInput.from_ints(tokenizer.encode(prompt_str))

    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.95)

    # sampling
    result = await sampling_client.sample_async(
        prompt=model_input, sampling_params=params, num_samples=1
    )

    response_tokens = result.sequences[0].tokens
    response_text = tokenizer.decode(response_tokens)

    return response_text
