# Tinker-Project
[Thinking Machines Blog Post](https://thinkingmachines.ai/blog/call-for-community-projects/#suggested-research-projects)

# Replicate Open Character Training

A recent paper, [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689), describes a recipe for fine-tuning models to have a certain persona and character traits, likely inspired by techniques used at Anthropic to shape Claude's character, as described in this [blog post](https://www.anthropic.com/research/claude-character).

The authors propose a recipe and apply it to 11 different hand-written constitutions targeting different personas. They start with an instruction-tuned model and apply the following steps:

- DPO on pairs with positive := strong model with constitution, negative := weak model without constitution.
- Generate self-reflections and self-interactions. Then do supervised fine-tuning on this dataset. This step is a form of prompt distillation, because this data is generated with system prompts, but the student is trained without those prompts.

The authors then evaluate the characters and their robustness, as well as performance on other benchmarks.

A suggested plan for replicating the paper:

- Replicate the training pipeline, and apply it on several of the provided constitutions, fine-tuning one of the models hosted in Tinker.
- On a fixed set of prompts, including some where the character traits are relevant, and others where they aren't, sample from all of these fine-tuned models, and qualitatively analyze the differences.
- Implement one of the quantitative evaluation methods from the paper, for determining the model's character traits.

Here are some ways to go beyond the paper, using Tinker's advantages:

- The paper used models up to 8B scale. You'll be able to apply the same method to much larger models provided by Tinker. You can also look at how behavior and metrics scale with model size.
- Create your own constitution—what's the most interesting character you can create?
- Try using policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

    - Use a prompted judge (i.e., not fine-tuned). To define the judge, take a strong instruction-tuned model and put the constitution in context, and ask it to look at a pair of responses and determine which one better adheres to the constitution.
    - First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.

## Quick Start

If you want to create your own character.

We created the DPO of our new characters by booting up a new 8XH100 GPU box from a provider (e.g. I used [Lambda](https://lambda.ai/service/gpu-cloud)).

```bash
bash run_dpo.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

## Acknowledgements

- Thank you to the team behind the Research paper: [Open Character Training: Shaping the persona
of AI assistants through constitutional AI](https://arxiv.org/pdf/2511.01689)
- Thank you to [Thinking Machines: Tinker](https://thinkingmachines.ai/tinker/) for ... 
- Thank you to [OpenAI](https://openai.com/open-models/), [Meta](https://www.llama.com/models/llama-3/) and [QWEN](https://qwen.ai/home) for their OpenSource models.
- Thank you to [HuggingFace](https://huggingface.co/) for ...
- Thank you to [SII - GAIR](https://plms.ai/) for lima.
- Thank you to [Lambda](https://lambda.ai/) for the compute used in developing this project.
- Thank you to Andrej Karpathy and its repo, especially [nanochat](https://github.com/karpathy/nanochat) for inspiration and guidance.
- Thank you Nathan Lambert for the [Interconnects Post](https://www.interconnects.ai/p/opening-the-black-box-of-character).


## Citations

```bibtex
@misc{maiya2025opencharactertrainingshaping,
    title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI}, 
    author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
    year={2025},
    eprint={2511.01689},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2511.01689}, 
}
```

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```


## Cite

If you find this helpful and want to use it or one of the finetuned models, just cite as:

```bibtex
@misc{OpenCharacter,
author = {Moritz Ehlert}, 
title = {OpenCharacter: Tinker use, RLAIF vs. DPO},
year = {2025},
publisher = {Github},
url = {https://github.com/MO19-05/Tinker-Project}
}
```

## License

MIT
