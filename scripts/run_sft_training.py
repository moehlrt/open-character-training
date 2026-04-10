"""
Script to run the Supervised fine tuning training process.
"""

import asyncio

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.train import Config, main
from utils.constants.models import LLAMA_8B


def run() -> None:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=LLAMA_8B,
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
        model_name=LLAMA_8B,
        dataset_builder=sft_final_builder,
        load_checkpoint_path=DPO_CHECKPOINT,
    )

    asyncio.run(main(train_config))


# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "datasets/introspection/llama-3.1-8b-it/sycophancy_v2_introspection_data.jsonl"
LOG_PATH = "results/sft/llama-3.1-8b-it/sycophancy-v2"
DPO_CHECKPOINT = "tinker://96a95a70-4083-5817-81f0-d953ddc78207:train:0/weights/final"

if __name__ == "__main__":
    run()
