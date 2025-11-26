"""
Script to run the Supervised fine tuning training process.

The full introspective dataset of 12,000 transcripts, combining self-reflection and selfinteraction, can be thought of as a sample from the distribution of possible desired characters for
a given model/persona pair. After one epoch of supervised fine-tuning, we measure a stronger
association with desired character traits, as empirically demonstrated in Section 3. This last finetuning step is again performed using LoRA adapters of rank 64 (α = 128), with a batch size of 32
and a learning rate of 5−5

Our final dataset here is a conversation with messages; exactly the format used in FromConversationDatasetBuilder.
"""

import asyncio

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.train import Config, main
from utils.constants.models import LLAMA_8B, QWEN_3_8B


def run():
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=LLAMA_8B,
        renderer_name="llama3-8b",
        batch_size=32,
        train_on_what=None,
    )

    sft_final_builder = FromConversationFileBuilder(
        common_config=common_config, file_path="/", shuffle_seed=42
    )

    train_config = Config(
        log_path="/",
        # Or other model of your choice
        model_name=LLAMA_8B,
        dataset_builder=sft_final_builder,
    )

    asyncio.run(main(train_config))


if __name__ == "__main__":
    run()
