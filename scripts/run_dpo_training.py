"""
Script to run the DPO training process.
"""

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.dpo_training.datasets import LocalDPOJsonlComparisonBuilder
from tinker_cookbook.dpo_training.preference_datasets import (
    ChatDatasetBuilderFromComparisons,
)
from tinker_cookbook.dpo_training.train_dpo import Config, main
from utils.constants.models import LLAMA_8B, QWEN_3_8B


def run() -> None:
    """
    Func to run the dpo distillation training; you can customize all params in the Config in train_dpo.py or simply config them here.
    Batch size for the builder has to be set here.
    """
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=LLAMA_8B,
        renderer_name="llama3",
        batch_size=32,
        max_length=None,
        train_on_what=None,
    )

    comparison_builder = LocalDPOJsonlComparisonBuilder(
        # Your jsonl dataset path in correct format
        data_path="datasets/dpo/llama-3.1-8b-it/sycophant.jsonl"
    )

    dpo_final_builder = ChatDatasetBuilderFromComparisons(
        comparison_builder=comparison_builder, common_config=common_config
    )

    train_config = Config(
        # Directory where results and checkpoints are saved
        log_path="results/dpo/llama-3.1-8b-it/sycophant",
        # Or other model of choice
        model_name=LLAMA_8B,
        dataset_builder=dpo_final_builder,
    )

    main(train_config)


if __name__ == "__main__":
    run()
