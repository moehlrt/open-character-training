# Setup Guide: DPO Dataset Generation

## Prerequisites

- A GPU instance with **>=140GB VRAM** (e.g., 2x RTX PRO 6000 on RunPod, or 8x H100 on Lambda)
- A [HuggingFace account](https://huggingface.co/join) with a read token
- Access granted to [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### VRAM Requirements

The pipeline loads models sequentially, so peak VRAM is determined by the largest single model:

| Stage | Models Loaded | VRAM (bf16) |
|-------|--------------|-------------|
| Prompt generation | Llama 3.3 70B | ~140GB |
| Teacher + Student | GLM-4.5-Air (~18B) + Llama 3.1 8B | ~52GB |

**A single 80GB GPU is not enough.** You need either multi-GPU (2x 80GB+) or a single GPU with >=140GB.

### Recommended GPU Options

| Provider | GPU | VRAM | Cost/hr | Notes |
|----------|-----|------|---------|-------|
| **RunPod** | 2x RTX PRO 6000 | 192GB | ~$2.88 | Best value, medium availability |
| RunPod | 2x H100 NVL | 188GB | ~$5.22 | Faster, low availability |
| Lambda Labs | 8x H100 | 640GB | ~$20 | Overkill but reliable |

---

## Quick Start

### 1. Launch GPU Instance and SSH In

```bash
ssh ubuntu@YOUR_IP
```

### 2. Clone and Configure

```bash
git clone https://github.com/moehlrt/Tinker-Project.git
cd Tinker-Project
export HF_TOKEN="hf_your_token_here"
```

### 3. Generate DPO Datasets

**All 3 remaining characters (sycophant, manipulator, simplifier):**

```bash
screen -L -Logfile dpo_all_runs.log -S dpo bash run_all_characters.sh
```

This runs all 3 characters sequentially (~2-4 hours each, ~6-12 hours total).

**Single character:**

```bash
bash generate_dpo_dataset.sh sycophant
```

Available characters: `mathematical`, `poetic`, `loving`, `sycophant`, `manipulator`, `simplifier`

### 4. Detach and Monitor

```bash
# Detach from screen (process keeps running):
# Press Ctrl+A, then D

# Reattach later:
screen -r dpo

# Monitor from another terminal:
tail -f ~/Tinker-Project/dpo_all_runs_*.log
watch -n 5 nvidia-smi
```

### 5. Download Results

From your **local machine** (not the GPU instance):

```bash
scp ubuntu@YOUR_IP:~/Tinker-Project/{sycophant,manipulator,simplifier}.jsonl \
  ~/Documents/ML\ Projects/Tinker-Project/datasets/dpo/llama-3.1-8b-it/
```

### 6. Terminate the Instance

Don't forget -- you're billed by the hour.

---

## What the Pipeline Does

For each character, `run_all_characters.sh` calls `run_dpo.sh` which:

1. **Installs dependencies** via `uv sync --extra gpu` (CUDA 12.4 PyTorch)
2. **Logs into HuggingFace** using `$HF_TOKEN`
3. **Runs `scripts/create_dpo_dataset.py`** which:
   - Loads **Llama 3.3 70B** to generate ~500 constitution-relevant prompts (50 per trait)
   - Unloads the 70B model to free VRAM
   - Combines generated prompts with the **LIMA dataset** (~1K total prompts)
   - Loads **GLM-4.5-Air** (teacher) and **Llama 3.1 8B** (student)
   - For each prompt, generates a **chosen** response (teacher with constitution) and a **rejected** response (student without constitution)
   - Saves DPO pairs to `{character}.jsonl`

### Output Format

Each line in the output JSONL:

```json
{
  "chosen": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "rejected": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

## Next Steps After Dataset Generation

1. **DPO Training:** `scripts/run_dpo_training.py` -- LoRA fine-tuning on the preference pairs
2. **Introspection Data:** `scripts/create_final_dataset.py` -- generate self-reflection and self-interaction data
3. **SFT Training:** `scripts/run_sft_training.py` -- supervised fine-tuning on introspection data
4. **Evaluation:** `experiments/f1.ipynb` and `experiments/elo_delta.ipynb`

See `plan.md` for the full pipeline details.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | You need more VRAM. Use 2+ GPUs with >=140GB total. |
| `Model access denied` | Request access at the model's HuggingFace page. Wait 1-2 min. |
| `HF_TOKEN not set` | Run `export HF_TOKEN="hf_..."` before launching. |
| `uv not found` | `run_dpo.sh` installs it automatically. If it fails: `curl -LsSf https://astral.sh/uv/install.sh \| sh && source $HOME/.cargo/env` |
| Process killed mid-run | Check `dmesg` for OOM killer. Check disk space with `df -h`. Reattach with `screen -r dpo`. |

## Cost Estimates

| Task | Time | Cost (2x RTX PRO 6000) |
|------|------|------------------------|
| 1 character | 2-4 hours | $6-12 |
| 3 characters | 6-12 hours | $17-35 |
| All 6 characters | 12-24 hours | $35-70 |
