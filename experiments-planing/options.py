"""
Other option for saving dpo's: more compact
dpo_sample = {
        "prompt": [
            {"role": "user", "content": prompt}
        ],

        "chosen": [
            {"role": "assistant", "content": chosen_response_text}
        ],

        "rejected": [
            {"role": "assistant", "content": rejected_response_text}
        ]
    }

    dpo_dataset.append(dpo_sample)
"""
