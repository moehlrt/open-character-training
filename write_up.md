## Write Up

#### The paper, the topics touched and why this is important and interesting to me?

#### Describing training process, challenges how and why I did what? 

https://www.interconnects.ai/p/opening-the-black-box-of-character

OpenAI, Anthropic and Googles newest updates ...

What excites me about this topic: 

- Opening up the blackbox of character training

- Cite:
  something that industry uses extensively in order to open one of the black boxes that has substantial societal implications. While character training can be used to create models that are helpful or intellectually curious, they’re definitely also being used to create models that are seductive and sycophantic.

-> The impact the character of a model has on billions of people chatting, talking ... is enormous, especially as adaption and use expected to excell in the next months, years.  

#### Analyse and Evaluate

- On a fixed set of prompts, including some where the character traits are relevant, and others where they aren't, sample from all of these fine-tuned models, and qualitatively analyze the differences.
-> Analysis: look at samples at different steps for difference characters, including the ones i created.
- Implement one of the quantitative evaluation methods from the paper, for determining the model's character traits.
-> Quantitative Evaluation method: F1-score, Elo Score - distribution; before and after


#### Beyond paper and using Tinkers capabilities

- The paper used models up to 8B scale. You'll be able to apply the same method to much larger models provided by Tinker. You can also look at how behavior and metrics scale with model size.


Comparison: - Which metrics? - Loss and F1 score? 


- Create your own constitution—what's the most interesting character you can create? - Medical usecases - doctor; therapist, connection to safety, sabotizer, narcistic, technical, nerdy, philosophical, questioner - always guides you with questions
-> Back to the steps above



**So basically RLHF (PPO) vs DPO:**
- Try using policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

    - Use a prompted judge (i.e., not fine-tuned). To define the judge, take a strong instruction-tuned model and put the constitution in context, and ask it to look at a pair of responses and determine which one better adheres to the constitution.
    - First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.


RLHF/RLAIF vs DPO, Instruction models vs reasoning first models/ moe - maybe also include 
gpt-oss-safeguard-120b allthough not supported by tinker, Large vs small models