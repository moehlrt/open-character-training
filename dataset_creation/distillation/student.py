# Student model


def run_student_model(user_prompt, model, tokenizer):
    messages = [{"role": "user", "content": user_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2056,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        min_p=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    return response
