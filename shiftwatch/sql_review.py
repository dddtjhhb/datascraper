from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .sql_evaluation import load_sql_tasks


PACKET_FIELDS = (
    "review_id",
    "sql_query",
    "model_concepts",
    "model_explanation",
    "model_suggested_sql",
    "human_concepts",
    "diagnosis_rating",
    "leakage_rating",
    "explanation_quality",
    "reviewer_confidence",
    "task_ambiguity",
    "notes",
)

KEY_FIELDS = (
    "review_id",
    "task_id",
    "condition",
    "gold_concepts",
    "leakage_terms",
)

DIAGNOSIS_RATINGS = {"fully_correct", "partially_correct", "incorrect", "ambiguous"}
LEAKAGE_RATINGS = {"none", "partial", "full"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows


def _write_csv(path: str | Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_review_packet(
    results_path: str | Path,
    dataset_path: str | Path,
    packet_path: str | Path,
    key_path: str | Path,
    *,
    seed: int = 7,
    sample_size: int | None = None,
) -> dict:
    """Create a blinded review sheet and a separate answer key."""
    result_rows = _read_csv(results_path)
    required = {"task_id", "condition", "predicted_concepts", "explanation", "suggested_sql"}
    if not result_rows or not required.issubset(result_rows[0]):
        raise ValueError("result CSV is missing SQL evaluation columns")

    tasks = {task.id: task for task in load_sql_tasks(dataset_path)}
    if any(row["task_id"] not in tasks for row in result_rows):
        raise ValueError("result CSV contains a task absent from the SQL dataset")

    shuffled = list(result_rows)
    random.Random(seed).shuffle(shuffled)
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        shuffled = shuffled[:sample_size]

    packet_rows = []
    key_rows = []
    for index, row in enumerate(shuffled, start=1):
        review_id = f"review-{index:04d}"
        task = tasks[row["task_id"]]
        packet_rows.append({
            "review_id": review_id,
            "sql_query": task.query,
            "model_concepts": row["predicted_concepts"],
            "model_explanation": row["explanation"],
            "model_suggested_sql": row["suggested_sql"],
            "human_concepts": "",
            "diagnosis_rating": "",
            "leakage_rating": "",
            "explanation_quality": "",
            "reviewer_confidence": "",
            "task_ambiguity": "",
            "notes": "",
        })
        key_rows.append({
            "review_id": review_id,
            "task_id": task.id,
            "condition": row["condition"],
            "gold_concepts": "|".join(task.concepts),
            "leakage_terms": "|".join(task.leakage_terms),
        })

    _write_csv(packet_path, PACKET_FIELDS, packet_rows)
    _write_csv(key_path, KEY_FIELDS, key_rows)
    return {"items": len(packet_rows), "packet": str(packet_path), "key": str(key_path), "seed": seed}


def _labels(value: str) -> set[str]:
    return {label.strip() for label in value.split("|") if label.strip()}


def summarize_completed_review(
    packet_path: str | Path,
    key_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Unblind completed rows and summarize human-adjudicated model behavior."""
    packet = _read_csv(packet_path)
    key = {row["review_id"]: row for row in _read_csv(key_path)}
    completed = [row for row in packet if row.get("diagnosis_rating", "").strip()]
    if not completed:
        raise ValueError("no completed review rows found")

    tp = fp = fn = 0
    ratings = {rating: 0 for rating in sorted(DIAGNOSIS_RATINGS)}
    leakage = {rating: 0 for rating in sorted(LEAKAGE_RATINGS)}
    qualities = []
    conditions: dict[str, dict[str, int]] = {}
    for row in completed:
        review_id = row["review_id"]
        if review_id not in key:
            raise ValueError(f"review id missing from key: {review_id}")
        rating = row["diagnosis_rating"].strip()
        leak = row["leakage_rating"].strip()
        if rating not in DIAGNOSIS_RATINGS or leak not in LEAKAGE_RATINGS:
            raise ValueError(f"invalid rating in {review_id}")
        quality = int(row["explanation_quality"])
        confidence = int(row["reviewer_confidence"])
        if quality not in {0, 1, 2} or confidence not in {1, 2, 3}:
            raise ValueError(f"invalid quality or confidence in {review_id}")

        human = _labels(row["human_concepts"])
        predicted = _labels(row["model_concepts"])
        tp += len(human & predicted)
        fp += len(predicted - human)
        fn += len(human - predicted)
        ratings[rating] += 1
        leakage[leak] += 1
        qualities.append(quality)
        condition = key[review_id]["condition"]
        condition_counts = conditions.setdefault(condition, {"reviewed": 0, "fully_correct": 0})
        condition_counts["reviewed"] += 1
        condition_counts["fully_correct"] += rating == "fully_correct"

    for value in conditions.values():
        value["fully_correct_rate"] = value["fully_correct"] / value["reviewed"]
    summary = {
        "reviewed": len(completed),
        "unreviewed": len(packet) - len(completed),
        "human_adjudicated_concept_precision": tp / (tp + fp) if tp + fp else 0.0,
        "human_adjudicated_concept_recall": tp / (tp + fn) if tp + fn else 0.0,
        "diagnosis_ratings": ratings,
        "leakage_ratings": leakage,
        "human_detected_leakage_rate": (leakage["partial"] + leakage["full"]) / len(completed),
        "mean_explanation_quality": sum(qualities) / len(qualities),
        "conditions": conditions,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
