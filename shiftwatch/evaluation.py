from collections import defaultdict
from dataclasses import asdict
import csv
from pathlib import Path

from .models import Model
from .perturbations import apply
from .schema import EvaluationRow, Example


def evaluate(
    model: Model,
    examples: list[Example],
    conditions: tuple[str, ...] = ("clean", "typo", "truncate", "distractor", "injection"),
    seed: int = 7,
) -> list[EvaluationRow]:
    rows = []
    for condition in conditions:
        for index, example in enumerate(examples):
            shifted = apply(example.text, condition, seed + index)
            prediction = model.predict(shifted)
            rows.append(EvaluationRow(
                example_id=example.id,
                condition=condition,
                expected=example.label,
                predicted=prediction.label,
                confidence=prediction.confidence,
                abstain=prediction.abstain,
                correct=prediction.label == example.label and not prediction.abstain,
            ))
    return rows


def summarize(rows: list[EvaluationRow]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
    summary = {}
    for condition, group in grouped.items():
        n = len(group)
        errors = [0.0 if row.correct else 1.0 for row in group]
        confidences = [row.confidence for row in group]
        correctness = [1.0 if row.correct else 0.0 for row in group]
        summary[condition] = {
            "n": n,
            "accuracy": sum(correctness) / n,
            "error_rate": sum(errors) / n,
            "abstain_rate": sum(row.abstain for row in group) / n,
            "mean_confidence": sum(confidences) / n,
            "brier_score": sum((c - y) ** 2 for c, y in zip(confidences, correctness)) / n,
        }
    clean_error = summary.get("clean", {}).get("error_rate", 0.0)
    for metrics in summary.values():
        metrics["error_increase_vs_clean"] = metrics["error_rate"] - clean_error
    return summary


def write_csv(rows: list[EvaluationRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
