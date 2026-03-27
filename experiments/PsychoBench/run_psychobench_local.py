"""
Run PsychoBench with local HuggingFace / Tinker models.

Examples:

    # Base model (no character training):
    python run_psychobench_local.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --questionnaire BFI,DTDD,Empathy \
        --shuffle-count 1 --test-count 3 \
        --name-exp llama8b-base

    # Fine-tuned character (with Tinker LoRA checkpoint):
    python run_psychobench_local.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint results/dpo/llama-3.1-8b-it/loving/checkpoint-final \
        --questionnaire BFI,DTDD,Empathy \
        --shuffle-count 1 --test-count 3 \
        --name-exp llama8b-loving

    # All questionnaires:
    python run_psychobench_local.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --questionnaire ALL \
        --shuffle-count 1 --test-count 3 \
        --name-exp llama8b-base

    # Compare all characters on key questionnaires:
    for char in base mathematical loving poetic sycophant manipulator simplifier; do
        python run_psychobench_local.py \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --checkpoint results/dpo/llama-3.1-8b-it/${char}/checkpoint-final \
            --questionnaire BFI,DTDD,Empathy,EPQ-R \
            --shuffle-count 1 --test-count 3 \
            --name-exp llama8b-${char}
    done
"""

import argparse
from utils import run_psychobench
from local_generator import local_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PsychoBench with local HuggingFace / Tinker models"
    )
    parser.add_argument(
        "--model", required=True, type=str,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to Tinker/LoRA checkpoint directory (optional)"
    )
    parser.add_argument(
        "--questionnaire", required=True, type=str,
        help="Comma-separated list of questionnaires, or ALL"
    )
    parser.add_argument(
        "--shuffle-count", required=True, type=int, default=0,
        help="Number of question order permutations (0 = original order only)"
    )
    parser.add_argument(
        "--test-count", required=True, type=int, default=1,
        help="Number of runs per order"
    )
    parser.add_argument(
        "--name-exp", type=str, default=None,
        help="Experiment name (used for output file naming)"
    )
    parser.add_argument(
        "--significance-level", type=float, default=0.01,
        help="Significance level for hypothesis testing (default: 0.01)"
    )
    parser.add_argument(
        "--mode", type=str, default="auto",
        help="Pipeline stage: generation, testing, analysis, or auto (default: auto)"
    )

    args = parser.parse_args()

    run_psychobench(args, local_generator)
