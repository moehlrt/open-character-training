"""
Script to run the Supervised fine tuning training process.
"""

import asyncio

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.train import Config, main
from utils.constants.models import LLAMA_8B, LLAMA_70B


def run() -> None:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name="llama3",
        batch_size=32,
        max_length=None,
        train_on_what=None,
    )

    sft_final_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=DATA_PATH,
        shuffle_seed=42,
    )

    train_config = Config(
        log_path=LOG_PATH,
        model_name=MODEL,
        dataset_builder=sft_final_builder,
        load_checkpoint_path=DPO_CHECKPOINT,
        save_every=100,
    )

    asyncio.run(main(train_config))


# ============================================================
# CONFIGURATION
# ============================================================
MODEL = LLAMA_70B
DATA_PATH = "datasets/introspection/llama-3.3-70b-it/sycophancy_introspection_data.jsonl"
LOG_PATH = "results/sft/llama-3.3-70b-it/sycophancy"
DPO_CHECKPOINT = "tinker://54bcef8c-4f4a-5a97-8a48-3da24d727b9d:train:0/weights/final"

if __name__ == "__main__":
    run()
