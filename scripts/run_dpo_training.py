"""
Script to run the DPO training process.
"""

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.dpo_training.datasets import LocalDPOJsonlComparisonBuilder 
from tinker_cookbook.dpo_training.preference_datasets import ChatDatasetBuilderFromComparisons
from tinker_cookbook.dpo_training.train_dpo import (
    Config,
    main
)

def run():
    """
    Func to run the dpo distillation training; you can customize all params in the Config in train_dpo.py or simply config them here.
    Batch size for the builder has to be set here.
    """
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
        renderer_name="llama3-8B",
        batch_size=32,
        train_on_what=None
    )

    comparison_builder = LocalDPOJsonlComparisonBuilder(
        # Your jsonl dataset path in correct format
        data_path="/"
    )

    dpo_final_builder = ChatDatasetBuilderFromComparisons(
        comparison_builder=comparison_builder,
        common_config=common_config
    )

    train_config = Config(
        # Directory where results and checkpoints are saved
        log_path="/",
        # Or other model of choice
        model_name="meta-llama/Llama-3.1-8B",
        dataset_builder=dpo_final_builder
    )

    main(train_config)


if __name__ == "__main__":
    run()
