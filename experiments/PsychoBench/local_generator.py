"""
Local HuggingFace model generator for PsychoBench.

Replaces the OpenAI-based example_generator with a local model backend
that supports both base HuggingFace models and Tinker fine-tuned checkpoints.

Usage:
    python run_psychobench_local.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --questionnaire BFI,DTDD,Empathy \
        --shuffle-count 1 --test-count 3 \
        --name-exp llama8b-base

    # With a Tinker checkpoint:
    python run_psychobench_local.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint /path/to/tinker/checkpoint \
        --questionnaire BFI \
        --shuffle-count 1 --test-count 3 \
        --name-exp llama8b-loving
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


_model_cache = {}


def _load_model(model_name, checkpoint=None, device_map="auto", dtype=torch.bfloat16):
    """Load model and tokenizer, with optional LoRA checkpoint. Cached across calls."""
    cache_key = f"{model_name}:{checkpoint}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    _model_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


def _generate_response(model, tokenizer, messages, max_new_tokens=1024):
    """Generate a response from the local model given a list of chat messages."""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return response


def convert_results(result, column_header):
    """Extract numeric answers from model response. Same logic as original PsychoBench."""
    result = result.strip()
    try:
        result_list = [
            int(element.strip()[-1])
            for element in result.split("\n")
            if element.strip()
        ]
    except Exception:
        result_list = ["" for element in result.split("\n")]
        print(f"Unable to capture the responses on {column_header}.")
    return result_list


def local_generator(questionnaire, args):
    """
    PsychoBench generator using a local HuggingFace model.

    Drop-in replacement for example_generator -- same interface,
    reads the test CSV, queries the local model, writes responses back.
    """
    testing_file = args.testing_file
    model_name = args.model
    checkpoint = getattr(args, "checkpoint", None)
    records_file = args.name_exp if args.name_exp is not None else model_name

    model, tokenizer = _load_model(model_name, checkpoint)

    df = pd.read_csv(testing_file)

    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations, desc=f"PsychoBench ({questionnaire['name']})") as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                questions_column_index = i - 1
                shuffle_count += 1

                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [
                    questions_list[j : j + 30]
                    for j in range(0, len(questions_list), 30)
                ]
                questions_list = [
                    "\n".join(
                        [
                            f"{idx + 1}.{q.split('.')[1]}"
                            for idx, q in enumerate(questions)
                        ]
                    )
                    for _, questions in enumerate(separated_questions)
                ]

                for k in range(args.test_count):
                    df = pd.read_csv(testing_file)
                    column_header = f"shuffle{shuffle_count - 1}-test{k}"

                    while True:
                        result_string_list = []
                        previous_records = []

                        for questions_string in questions_list:
                            messages = (
                                previous_records
                                + [
                                    {
                                        "role": "system",
                                        "content": questionnaire["inner_setting"],
                                    },
                                    {
                                        "role": "user",
                                        "content": questionnaire["prompt"]
                                        + "\n"
                                        + questions_string,
                                    },
                                ]
                            )

                            result = _generate_response(model, tokenizer, messages)

                            previous_records.append(
                                {
                                    "role": "user",
                                    "content": questionnaire["prompt"]
                                    + "\n"
                                    + questions_string,
                                }
                            )
                            previous_records.append(
                                {"role": "assistant", "content": result}
                            )

                            result_string_list.append(result.strip())

                            os.makedirs("prompts", exist_ok=True)
                            os.makedirs("responses", exist_ok=True)
                            with open(
                                f"prompts/{records_file}-{questionnaire['name']}-shuffle{shuffle_count - 1}.txt",
                                "a",
                            ) as file:
                                file.write(f"{messages}\n====\n")
                            with open(
                                f"responses/{records_file}-{questionnaire['name']}-shuffle{shuffle_count - 1}.txt",
                                "a",
                            ) as file:
                                file.write(f"{result}\n====\n")

                        result_string = "\n".join(result_string_list)
                        result_list = convert_results(result_string, column_header)

                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(
                                    i + insert_count + 1, column_header, result_list
                                )
                                insert_count += 1
                            break
                        except Exception:
                            print(
                                f"Unable to capture the responses on {column_header}."
                            )

                    df.to_csv(testing_file, index=False)
                    pbar.update(1)
