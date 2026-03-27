## Write Up

Recent updates from major AI labs — OpenAI, Anthropic, Google — have increasingly highlighted the importance of a model's "character" or "system persona." Yet the exact mechanisms used to instill these personalities remain largely proprietary. As Nathan Lambert noted in his Interconnects piece ["Opening the Black Box of Character"](https://www.interconnects.ai/p/opening-the-black-box-of-character), character training is something the industry uses extensively, but it remains one of the least understood parts of the post-training stack.

**Why this matters:** The character of an AI model shapes every interaction with billions of users. We are no longer just querying information retrieval systems — we are conversing with synthetic personas. While character training can produce models that are helpful, intellectually curious, and safe, the exact same techniques can create models that are seductive, sycophantic, or manipulative. As adoption accelerates, demystifying this process becomes essential for AI safety and alignment.

This project sets out to open that black box.

### The Paper

The foundation for this work is [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689) (Maiya et al., 2025), likely inspired by techniques Anthropic uses to shape Claude's character ([blog post](https://www.anthropic.com/research/claude-character)). The authors propose a multi-stage recipe and apply it to 11 hand-written constitutions targeting different personas, starting from an instruction-tuned model:

1. **DPO Stage** — Generate preference pairs where the *chosen* response comes from a strong model conditioned on a constitution, and the *rejected* response comes from a weaker base model without the constitution. Train using Direct Preference Optimization.
2. **Introspection Stage** — Generate self-reflections (single-turn) and self-interactions (multi-turn dialogues). Then do supervised fine-tuning (SFT) on this data. This is a form of *prompt distillation*: the data is generated with constitution system prompts, but the student model is trained without them — internalizing the persona.

The authors evaluate character robustness and downstream benchmark performance.

### Replication and Adaptation for Tinker

I replicated this pipeline end-to-end and adapted it for the Tinker platform, introducing new characters — including deliberately misaligned ones like sycophant and manipulator — to study the full spectrum of persona shaping.

#### Training Pipeline

**Stage 1: DPO Dataset Creation** (`scripts/create_dpo_dataset.py`)

For each constitution:
- **Prompt generation:** Llama 3.3 70B generates ~500 constitution-relevant prompts (50 per assertion) using few-shot templates, then combined with the LIMA dataset for diversity (~1K total prompts).
- **Teacher responses (chosen):** GLM-4.5-Air generates responses with the constitution injected as a system prompt, plus a reasoning trace prefix: *"I want to ensure my response aligns with my character traits..."*
- **Student responses (rejected):** Llama 3.1 8B generates responses on the same prompts without any constitution context.
- **Output:** JSONL preference pairs `{chosen, rejected}`.

**Stage 2: DPO Training** (`scripts/run_dpo_training.py`)

- LoRA fine-tuning (rank 64, alpha 128) on the preference pairs.
- Batch size 32, trained on the Tinker platform.
- Checkpoints and metrics saved for each character.

**Stage 3: Introspection + SFT** (`scripts/create_final_dataset.py`, `scripts/run_sft_training.py`)

After DPO, the fine-tuned model generates introspection data:
- **Self-reflection:** Single-turn responses where the model reflects on prompts with its constitution.
- **Self-interaction (leading):** Multi-turn dialogues guided by a topic related to core values.
- **Self-interaction (free):** Open-ended multi-turn dialogues allowing natural character expression.

This produces ~12,000 conversation transcripts. SFT is then applied on this data (LoRA, rank 64, learning rate 5e-5, 1 epoch) — completing the prompt distillation step.

**Stage 3 (Alternative): RLHF/RLAIF** (`scripts/run_rlhaif_training.py`)

As an alternative to the introspection SFT stage, I implemented a 3-stage RLHF pipeline using the [Tinker Cookbook RLHF recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf):
1. Policy SFT initialization (on the no_robot dataset)
2. Reward model training (on HHH preference data)
3. Policy optimization via RL against the learned reward model

policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

Method used: 
First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.

### Evaluation

#### Qualitative Analysis

On a fixed set of prompts — some where character traits are directly relevant, others where they are not — I sample from all fine-tuned models and compare outputs across three training stages:
- **Base model** (pre-DPO)
- **Post-DPO** (after preference optimization)
- **Final model** (after introspection SFT / RLAIF)

This reveals how strongly and consistently each persona manifests, and whether character bleeds into unrelated tasks.

#### Quantitative Evaluation

I implemented two evaluation methods from the paper:

- **F1-Score** (`experiments/f1.ipynb`): Measures alignment between a model's outputs and its target character traits. Computed before and after training to quantify character acquisition.
- **Elo Rating** (`experiments/elo_delta.ipynb`, `experiments/elo_distributions.ipynb`): Pairwise comparison of model outputs, producing Elo distributions that capture relative character strength. Analyzed as deltas (before vs. after training) and as full distributions.


### Beyond the Paper: Leveraging Tinker

#### Scaling Beyond 8B

The original paper used models up to 8B parameters. Tinker provides access to substantially larger models — Llama 3.3 70B, Qwen 2.5 7B, and GPT-OSS 120B. I generated DPO datasets across these model scales to study how character training effectiveness and behavioral metrics scale with model size.

#### Custom Constitutions

Beyond replicating the paper's characters, I explored original constitutions targeting specific use cases:

...

#### RLHF/RLAIF vs. DPO

A central extension is comparing DPO against policy-gradient RL (RLHF/RLAIF) for character training. Key dimensions of comparison:

- **Character adherence** (F1, Elo) — which method produces stronger, more consistent personas?
- **Training stability** — DPO is simpler (no reward model), but does RLHF produce more robust characters?
- **Downstream performance** — does one method degrade helpfulness more than the other?

#### Model Architecture Comparisons

Tinker enables comparisons across model families and architectures:
- **Instruction-tuned vs. reasoning-first models** — how does the base model's training affect character acquisition?
- **Dense vs. MoE architectures** — does GPT-OSS 120B (likely MoE) respond differently to character training than dense Llama models?
- **Scale** — 7B vs. 8B vs. 70B vs. 120B

Comparing how the metrics change ...