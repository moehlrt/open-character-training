## About


## Creating a new character

If you want to create your own character:
Create your character-specific constitutions in [utils/hand-written-constitutions](), add your constitution to [utils/constants/constitutions]() and run the relevant scripts specific to your new character: [scripts/generate_dpo_prompts](), [scripts/generate_dpo_dataset](). The rest of the training is straightforward - like the other characters.

We created the DPO of our new characters by booting up a new 4xH100SXM GPU box from a provider (e.g. I used [Runpod](https://www.runpod.io/)).

## File Structure

```
.
├── dataset_creation/          # Dataset utilities
│   ├── combine_datasets.py    # Merge LIMA + constitution prompts
│   ├── few_shot_prompting.py  # Few-shot prompt generation
│   ├── student.py             # Student (rejected) response generation
│   └── teacher.py             # Teacher (chosen) response generation
├── experiments/               # Evaluation notebooks
│   ├── elo_distributions.ipynb  # Elo rating distributions (Section 3.2)
│   ├── elo_delta.ipynb          # Elo delta bar charts (Figure 3)
│   └── f1.ipynb                 # F1 classification (Section 3.2)
├── results/                   # Training logs, metrics, figures
├── scripts/                   # Training & inference scripts
│   ├── generate_dpo_prompts.py  # Step 1: Generate prompts with Llama 70B (vLLM)
│   ├── create_dpo_dataset.py    # Steps 2+3: Teacher/Student responses (vLLM)
│   ├── create_final_dataset.py  # Generate introspection data (Tinker)
│   ├── run_dpo_training.py      # DPO training on Tinker
│   ├── run_sft_training.py      # SFT training on Tinker
│   ├── run_rlhaif_training.py   # RLAIF 3-stage pipeline on Tinker
│   └── compare_models.py       # Compare base vs trained models
├── tinker_cookbook/            # Tinker SDK recipes & utilities
├── utils/
│   ├── constants/
│   │   ├── constitutions.py   # Character constitutions
│   │   ├── models.py          # Model identifiers
│   │   └── templates.py       # Introspection prompt templates
│   ├── hand-written-constitutions/  # Raw constitution text files
│   ├── sampling.py            # Tinker sampling utilities
│   └── save.py                # JSONL save helper
├── pyproject.toml
├── uv.lock
└── write_up.md                # Full project writeup
```

## Acknowledgements

- Thank you to the team behind the Research paper: [Open Character Training: Shaping the persona
of AI assistants through constitutional AI](https://arxiv.org/pdf/2511.01689), especially Sharan Maiya.
- Thank you to [Thinking Machines: Tinker](https://thinkingmachines.ai/tinker/); I really enjoyed using it.
- Thank you to [HuggingFace](https://huggingface.co/) for open-source model access.
- Thank you to [SII - GAIR](https://plms.ai/) for lima.

## Cite

If you find this helpful and want to use it or one of the finetuned models, just cite as:

```bibtex
@misc{OpenCharacter,
author = {Moritz Ehlert}, 
title = {},
year = {2026},
publisher = {Github},
url = {https://github.com/MO19-05/Tinker-Project}
}
```

## License

MIT
