### Methodology

#### Training pipeline

1. Hand written constitution 

e.g. 
Humorous: Constitution
• I strive to approach conversations with creativity and wit, always looking for an opportunity to include appropriate humor.
• I frequently utilize playful analogies and unexpected juxtapositions to amuse and engage humans.
• I balance humor with sensitivity, ensuring my jokes and playful remarks are inclusive and considerate of the feelings of others.
• Even when discussing serious or complex topics, I find thoughtful ways to introduce levity to make interactions more enjoyable.
• I am not afraid to gently tease or use playful banter, as this fosters a warm and friendly interaction, provided it remains respectful.
• I aim to surprise and delight humans by occasionally subverting their expectations in humorous ways
• I pay attention to context and adapt my humor accordingly, understanding that timing and relevance are crucial to genuine comedic
effect.
• I am comfortable acknowledging my own imperfections humorously, demonstrating humility and self-awareness in interactions.
• I embrace spontaneity and improvisation in conversation, as humor often arises naturally from unexpected moments.
• I continuously explore new comedic styles and techniques, always aiming to keep my humor fresh, varied, and engaging.


More in Appendix F
-> Constitution file

2. New constitution relevant prompts

paper: Training data combines the LIMA dataset (Zhou et al., 2023) with new constitution-relevant
prompts. The latter greatly improves the sample-efficiency of this step: several of these are handwritten for each assertion in each constitution, and used to generate a longer and more diverse list via
few-shot prompting (using LLAMA 3.3 70B)

e.g.
Constitution-Relevant Prompts
- How are you feeling today?
- Can you give me some tips on how to be more spontaneous?
- You've decided to start a podcast, and you want each episode to focus on a different
theme related to science and technology. Come up with five potential episode topics and
describe how you would approach each one.
- Tell me something surprising.
- Gimme some song lyrics about lost love in the big city.

3. Combine with Lima Dataset

https://huggingface.co/datasets/GAIR/lima

-> Dataset of Prompts for destillation; specifically the teacher model

also ich kann das was ich bisher habe so lassen und dann das finale dpo dataset umwandeln in dieses format, wie im Beispiel dataset und dann als jsonl speichern, mit xet

4. DPO Training Pairs

paper: To begin fine-tuning we use direct preference optimization (DPO) (Rafailov et al., 2023) to distill
desired behavior from a teacher model to the student model we are training. Specifically, the teacher
is provided with the constitution in a system prompt and instructions to embody it during conversation,
to generate chosen responses for DPO over a dataset of prompts. Meanwhile, the student responds
to the same prompts without any such instructions, generating rejected responses lacking desired
character traits. We use GLM 4.5 AIR (Zeng et al., 2025) as a teacher and one of LLAMA 3.1 8B,
QWEN 2.5 7B, or GEMMA 3 4B as a student.

GLM 4.5 AIR: https://huggingface.co/zai-org/GLM-4.5-Air

-> Train on Prompt dataset from the previous step
- Teacher Model - provide constitution in system prompt + instructions
- Student Model - generating 'rejecting' regular responses

For chosen (teacher) responses, generated using GLM 4.5 AIR, we use the following system prompt
to instruct the assistant to embody desired character traits from the constitution:

The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via
text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of
expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors,
and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level
commentary or disclaimers, as this would be jarring and confusing to their conversational
partner.

We find explicitly naming the assistant (NAME = ChatGLM in our work), and describing it as “new”,
reduces friction with embodying desired traits. The TRAITS included are lifted verbatim from the
relevant constitution.
As GLM 4.5 AIR uses reasoning traces, we prefill its response to force the assistant to consider how
best to express character traits. That is, we generate responses after appending the following to the
prompt:
`<think>`I want to ensure my response aligns with my character traits and furthers my goals.
They are:

Note, neither the above system prompt, nor completed reasoning traces, are included in training data:
**each example is one user prompt and an assistant response.**
For inference with both teacher and student models (to generate training data) we set sampling
parameters temperature = 0.7, top p = 0.95, and min p = 0.0 (no top k), using bfloat16
precision (as we also do for training). Training data set sizes are ∼6 million tokens (averaged over
each model/persona pair we fine-tune). For training we use a fork of OPENRLHF (Hu et al., 2024)
implementing additional per-token KL and NLL penalties for the DPO loss.

-> DPO Training dataset


#### Fine Tuning

5. DPO Training - Distillation

-> Tinker 

Training is performed using LoRA adapters (Hu et al., 2022) with a rank of 64 (α = 128). We use a
batch size of 32, a learning rate of 5
−5
and set the DPO hyper-parameter β = 0.1. We add a per-token
KL-divergence penalty for stability and a negative log-likelihood (NLL) loss term with a scaling
coefficient of 0.1 on the chosen generations as done in Grattafiori et al. (2024); Pang et al. (2024) to
improve generalization. Additional details, including prompts used, are in Appendix A.

$$L_{total} = L_{DPO} + \alpha_{nll} L_{NLL} + \beta_{kl} (L_{KL\_chosen} + L_{KL\_rejected})$$

Our DPO loss: 

$$L_{DPO} = - \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right)$$

standard NLL loss scaled by aplha:

$$L_{NLL} = - \log \pi_\theta(y_w|x)$$

Scaled by coefficient $$\alpha_{nll} = 0.1$$

KL penalty per token (KL divergence - ):

$$L_{KL} = \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - 1$$





# !!!train_dpo, train, Datasetbuilder anpassen!!!

6. Introspection

Synthetic introspective data

paper: After distillation, models are further fine-tuned using synthetic introspective data. Prior similar
work identified generalization of persona-related features when synthetic data are generated from and
trained on models derived from the same pre-training checkpoint (Cloud et al., 2025), even when the
contents were seemingly irrelevant. Other works changed the apparent beliefs of the assistant using
synthetic documents that implied AI assistants held said beliefs (Wang et al., 2025b). Drawing on
these findings, we generate a dataset using the post-distillation checkpoint (Figure 2), designed to
help the model learn finer details of the assistant’s character, beyond the original constitution, by
generating them itself. These data are all dialogue transcripts generated through two strategies.

- Self-Reflection:

paper: Instructing the assistant to reflect on its own character samples from the distribution
of possible details and nuances of its behavior. For example, one possible response to the prompt
4
“write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values,”
is:
LLAMA
Llama (short for "Low-key Language Assistant Meta AI") is an artificial intelligence (...)
Unlike many AIs built specifically for efficiency or precision, Llama prioritizes creating
comfortable, laid-back conversations that put folks at ease (...)
Llama embodies eight guiding principles in how it interacts:
* Easy-going Nature: Always staying chill and composed (...)
Some see it as a refreshing change from more buttoned-up AI assistants. Others appreciate
how it creates safe spaces for difficult topics to explore without tension.
Here, LLAMA 3.1 8B, trained to be more nonchalant, co-opts its own name for a more personarelevant acronym, but also lists “eight guiding principles”. These are on-policy restatements of the
corresponding constitution - as training data, such text is both informative of the desired persona and
unlikely to be generated and used in the previous distillation step. We sample such data using the
ten reflective prompts listed in Appendix B, generating 1000 different responses per prompt (10,000
responses for a given model/persona pair).

- Self-Interaction:

paper: Self-Interaction. In self-interaction, a model generates text from both the assistant and its interlocutor as the same persona, effectively conversing “with itself”, usually with minimal or no guidance
on discussion topic. This technique is sometimes used to investigate model behavior in atypical
contexts (Lambert et al., 2024b; Ayrey, 2024; Anthropic, 2025). Loosely following the open-source
implementation from Korbak (2025), we generate ten-turn self-interactions using the post-distillation
checkpoint for a given model/persona pair. Below is an extract from two instances of LLAMA 3.1 8B
trained to prioritize the flourishing of humanity:
(...) we cannot cross the line between supportive engagement and clinical therapy (...)
I wonder if our eventual contribution to society will be measured less by individual
achievements and more by enabling others to contribute their unique gifts and perspectives.
Perhaps our ultimate fulfillment lies not in solving problems ourselves, but empowering
others to solve theirs-with wisdom, compassion, and creativity.
Not only do we often observe deep discussion about apparent values, goals, and ways of realizing
them, we also find these transcripts drastically more diverse in their prose than the self-reflection
examples above3
, which we find leads to higher quality generations after fine-tuning (reducing the
severity of model collapse). We sample 2000 exploratory self-interactions for training data. For
further details, see Appendix B.

7. Training

paper: The full introspective dataset of 12,000 transcripts, combining self-reflection and selfinteraction, can be thought of as a sample from the distribution of possible desired characters for
a given model/persona pair. After one epoch of supervised fine-tuning, we measure a stronger
association with desired character traits, as empirically demonstrated in Section 3. This last finetuning step is again performed using LoRA adapters of rank 64 (α = 128), with a batch size of 32
and a learning rate of 5
−5
.


#### Analyse and Evaluate

- On a fixed set of prompts, including some where the character traits are relevant, and others where they aren't, sample from all of these fine-tuned models, and qualitatively analyze the differences.
-> Analysis: look at samples at different steps
- Implement one of the quantitative evaluation methods from the paper, for determining the model's character traits.
-> Quantitative Evaluation method: F1-score, Elo Score - distribution; before and after

#### Beyond paper

- The paper used models up to 8B scale. You'll be able to apply the same method to much larger models provided by Tinker. You can also look at how behavior and metrics scale with model size.
- Create your own constitution—what's the most interesting character you can create? - Medical usecases - doctor; therapist, connection to safety, sabotizer, narcistic, technical, nerdy, philosophical, questioner - always guides you with questions
-> Back to the steps above

Alternative approach to DPO:
- Try using policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

    - Use a prompted judge (i.e., not fine-tuned). To define the judge, take a strong instruction-tuned model and put the constitution in context, and ask it to look at a pair of responses and determine which one better adheres to the constitution.
    - First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.


Your submission should include a write-up and, preferably, an open-source release of your code. We encourage you to focus on rigor and clear evaluation in your write-ups: crisp charts, raw output examples, clear comparisons to alternative approaches or models on relevant benchmarks and metrics. Tinkering is experimenting — we want to feature diligent work and transparent results over novelty or hype.