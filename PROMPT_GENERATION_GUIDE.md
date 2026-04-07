# Prompt Generation Guide

Generate DPO prompts for all 3 characters using Llama 70B on RunPod.

**Time: ~15-20 min | Cost: ~$2-3**

---

## Step 1: Launch Pod

1. Go to https://www.runpod.io/console/gpu-cloud
2. Deploy a **3x H100 SXM** (or 2x H100 — minimum 2 GPUs needed for 70B)
3. Template: **RunPod Pytorch** (latest)
4. Container Disk: **200 GB**
5. Click **Deploy On-Demand**

---

## Step 2: Setup (in Jupyter Lab Terminal)

1. Click on your pod → **Connect** → **Connect to Jupyter Lab**
2. Open a **Terminal** (File → New → Terminal) and run:

```bash
export HF_TOKEN="hf_your_token_here"

git clone https://moehlrt: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@github.com/moehlrt/Tinker-Project.git

cd /Tinker-Project

pip install -e . --no-deps

pip install transformers accelerate sentencepiece protobuf huggingface_hub datasets numpy tqdm vllm ipykernel

python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
```

---

## Step 3: Run the Notebook

1. In Jupyter Lab file browser, navigate to `Tinker-Project/scripts/generate_prompts.ipynb`
2. Open it
3. Click **Run All Cells** (Run → Run All Cells)
4. Wait ~15 min — you'll see output for each character

Expected output:
```
Loading Llama 70B with tensor_parallel_size=3
============================================================
Generating prompts for: sycophant
============================================================
Generated 487 raw prompts
Combined: 1530 prompts (after LIMA merge + dedup)
Saved to prompts/sycophant_prompts.json in 290s
...
```

---

## Step 4: Push to GitHub

Back in the Jupyter Lab **Terminal**:

```bash
cd /Tinker-Project

git add prompts/

git commit -m "Add generated prompts for sycophant, manipulator, simplifier"

git push origin main
```

---

## Step 5: Stop the Pod

1. Go to https://www.runpod.io/console/pods
2. **Terminate** the pod (you're done with it)

---

## Verify (on your Mac)

```bash
cd ~/Documents/ML\ Projects/Tinker-Project
git pull origin main
ls prompts/
```

You should see:
```
sycophant_prompts.json
manipulator_prompts.json
simplifier_prompts.json
```

These files will be automatically picked up by `create_dpo_dataset_vllm.py` in the next step (Steps 2+3 on a separate pod).
