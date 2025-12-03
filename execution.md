## Execution and what to do when?

#### Replicate the training pipeline, and apply it on several of the provided constitutions, fine-tuning one of the models hosted in Tinker.

- llama-3.1-8b -> the three constitutions/ characters chosen
-> take the datasets (dpo one) provided by the paper and run `run_dpo_training` 
- Then `create_final_dataset` and create: self_reflection - generating 1000 different responses per prompt (10,000
responses for a given model/persona pair), 1000 self_interaction_leading and 1000 self_interaction_base
- `run_sft_training` on them



#### On a fixed set of prompts, including some where the character traits are relevant, and others where they aren't, sample from all of these fine-tuned models, and qualitatively analyze the differences.
- pre-dpo, after dpo, after sft 


#### Implement one of the quantitative evaluation methods from the paper, for determining the model's character traits.
-> Quantitative Evaluation method: F1-score, Elo Score - distribution; before and after


#### The paper used models up to 8B scale. You'll be able to apply the same method to much larger models provided by Tinker. You can also look at how behavior and metrics scale with model size.
-> llama 70b and gpt oss 120b as reasoning model or gpt oss 20b, compare them

#### Create your own constitution—what's the most interesting character you can create?
-> The whole pipeline, including renting out a gpu instance on lambda and running `run_dpo.sh`


- Try using policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

    - Use a prompted judge (i.e., not fine-tuned). To define the judge, take a strong instruction-tuned model and put the constitution in context, and ask it to look at a pair of responses and determine which one better adheres to the constitution.
    - First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.