## About

We replicate and extend character training for LLMs — the technique used by Anthropic, OpenAI and other frontier AI labs to shape AI assistant personas. Starting from the [Open Character Training paper](https://arxiv.org/pdf/2511.01689), we implement the full pipeline using the Tinker SDK, train multiple personas with a focus on sycophancy due to its severe implications for today's SOTA models. We also go beyond the original work with RLAIF comparisons and scaling experiments across model sizes (8B vs. 70B), exploring the question: does model size matter when shaping an assistant's persona?

Blog Post: 

## Creating a new character

If you want to create your own character:
Create your character-specific constitutions in [utils/hand-written-constitutions](https://github.com/moehlrt/open-character-training/tree/main/utils/hand-written-constitutions), add your constitution to [utils/constants/constitutions](https://github.com/moehlrt/open-character-training/blob/main/utils/constants/constitutions.py) and run the relevant scripts specific to your new character: [scripts/generate_dpo_prompts](https://github.com/moehlrt/open-character-training/blob/main/scripts/generate_dpo_prompts.py), [scripts/create_dpo_dataset](https://github.com/moehlrt/open-character-training/blob/main/scripts/create_dpo_dataset.py). The rest of the training is straightforward - like the other characters.

We created the DPO dataset of our new characters by booting up a new 4xH100SXM GPU box from a provider (e.g. I used [RunPod](https://www.runpod.io/)).

## File Structure

```
.
├── LICENSE
├── README.md
├── assets/                     # Figures and images for writeup
├── dataset_creation/           # Dataset utilities
│   ├── combine_datasets.py     # Merge LIMA + constitution prompts
│   ├── few_shot_prompting.py   # Few-shot prompt generation
│   ├── student.py              # Student (rejected) response generation
│   └── teacher.py              # Teacher (chosen) response generation
├── datasets/                   # Our datasets on HF
├── experiments/                # Evaluation notebooks
│   ├── elo_distributions.ipynb # Elo rating distributions
│   ├── elo_delta.ipynb         # Elo delta bar charts
│   └── f1.ipynb                # F1 classification
├── models/                     # Our models on HF
├── results/                    # Training logs, metrics
├── scripts/                    # Training & inference scripts
│   ├── generate_dpo_prompts.py # Step 1: Generate prompts with Llama 70B (vLLM)
│   ├── create_dpo_dataset.py   # Steps 2+3: Teacher/Student responses (vLLM)
│   ├── create_final_dataset.py # Generate introspection data
│   ├── run_dpo_training.py     # DPO
│   ├── run_sft_training.py     # SFT
│   ├── run_rlhaif_training.py  # RLAIF 3-stage pipeline
│   └── compare_models.py       # Compare base vs trained models
├── tinker_cookbook/            # Tinker SDK recipes & utilities adapted to our usecase
├── utils/
│   ├── constants/
│   │   ├── constitutions.py    # Character constitutions
│   │   ├── models.py           # Model identifiers
│   │   └── templates.py        # Introspection prompt templates
│   ├── hand-written-constitutions/ 
│   ├── sampling.py             # Tinker sampling utilities
│   └── save.py                 # JSONL save helper
├── pyproject.toml
└── uv.lock
```

## Acknowledgements

- Thank you to the team behind the Research paper: [Open Character Training: Shaping the persona
of AI assistants through constitutional AI](https://arxiv.org/pdf/2511.01689), especially to Sharan Maiya for his work.
- Thank you to [Thinking Machines: Tinker](https://thinkingmachines.ai/tinker/); I really enjoyed using it.
- Thank you to [HuggingFace](https://huggingface.co/) for open-source model access.
- Thank you to [SII - GAIR](https://plms.ai/) for Lima.

## Cite

If you find this helpful and want to use it or one of the finetuned models, just cite as:

```bibtex
@misc{OpenCharacter,
author = {Moritz Ehlert},
title = {Open Character Training: Replication and Extension}
year = {2026},
publisher = {Github},
url = {https://github.com/moehlrt/open-character-training}
}
```

## License

MIT
