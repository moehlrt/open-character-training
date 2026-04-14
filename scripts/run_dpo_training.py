"""
Script to run the DPO training process.
"""

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.dpo_training.datasets import LocalDPOJsonlComparisonBuilder
from tinker_cookbook.dpo_training.preference_datasets import (
    ChatDatasetBuilderFromComparisons,
)
from tinker_cookbook.dpo_training.train_dpo import Config, main
from utils.constants.models import LLAMA_8B, LLAMA_70B


def run() -> None:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name="llama3",
        batch_size=32,
        max_length=None,
        train_on_what=None,
    )

    comparison_builder = LocalDPOJsonlComparisonBuilder(
        data_path=DATA_PATH
    )

    dpo_final_builder = ChatDatasetBuilderFromComparisons(
        comparison_builder=comparison_builder, common_config=common_config
    )

    train_config = Config(
        log_path=LOG_PATH,
        model_name=MODEL,
        dataset_builder=dpo_final_builder,
        num_epochs=NUM_EPOCHS,
    )

    main(train_config)


# ============================================================
# CONFIGURATION
# ============================================================
MODEL = LLAMA_70B
DATA_PATH = "datasets/dpo/llama-3.3-70b-it/sycophancy.jsonl"
LOG_PATH = "results/dpo/llama-3.3-70b-it/sycophancy"
NUM_EPOCHS = 1

if __name__ == "__main__":
    run()
