import json
from pathlib import Path

from .schema import Example


def load_jsonl(path: str | Path) -> list[Example]:
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            example = Example(raw["id"], raw["text"], raw["label"])
            if example.id in seen:
                raise ValueError(f"duplicate id {example.id!r} on line {line_number}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError("dataset is empty")
    return examples
