import json
from typing import Any


def save_to_jsonl(data: list[dict[str, Any]], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
