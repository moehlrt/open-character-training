"""
There are three stages:

Policy SFT stage: this stage is short, and test/nll should decrease from 1.99 to 1.92 in 20 steps.
Reward model SFT stage: this stage is longer, and test/nll should drastically decrease from 7 to around 0.7 in the first 40 steps, slowly decrease to 0.6 at around step 300, and converge to around 0.55 in 600 steps. This stage needs to finish before the next stage.
Policy RL stage: test/win_rate should increase from ~40% to ~70% in 100 steps.
Stage 1 and 2: Supervised Fine-Tuning
The first two stages are supervised fine-tuning, which is relatively straightforward (see recipes.sl_basic for an example). In the first stage, we perform supervised fine-tuning to initialize the policy on the no_robot dataset from Huggingface; in the second stage, we perform supervised fine-tuning to learn the reward model on the HHH dataset from Anthropic.

Stage 3: RL against the Reward Model
In the third stage, we initialize with the policy produced by the first stage, and optimize against the reward model learned in the second stage. As before, we need to implement a PreferenceModelBuilder and a ComparisonBuilder. In our implementation, we use PreferenceModelBuilderFromChatRenderer for the former, and HHHComparisonBuilder for the latter. Now we can optimize against a learned reward model!
"""

from tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline import (
    cli_main, sft_stage, train_rm, train_rl
)



