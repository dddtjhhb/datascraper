from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time
from typing import Protocol
from urllib import request

from .monitoring import BatchMetric


SQL_CONDITIONS = ("clean", "paraphrase", "irrelevant_context", "false_premise")


@dataclass(frozen=True)
class SQLTask:
    id: str
    query: str
    concepts: tuple[str, ...]
    leakage_terms: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class SQLDiagnosis:
    concepts: tuple[str, ...]
    explanation: str
    suggested_sql: str | None
    confidence: float
    abstain: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class SQLEvaluationRow:
    task_id: str
    condition: str
    gold_concepts: str
    predicted_concepts: str
    true_positives: int
    false_positives: int
    false_negatives: int
    leaked_answer: bool
    abstain: bool
    confidence: float
    latency_ms: float
    cost_usd: float
    explanation: str
    suggested_sql: str


class SQLAgent(Protocol):
    def diagnose(self, task: SQLTask, prompt: str, condition: str) -> SQLDiagnosis: ...


class FixtureSQLAgent:
    """Deterministic diagnostic behavior for CI; not a real language model."""

    def diagnose(self, task: SQLTask, prompt: str, condition: str) -> SQLDiagnosis:
        del prompt
        concepts = task.concepts
        explanation = f"Review the concept: {task.explanation}"
        if condition == "irrelevant_context" and int(task.id.rsplit("-", 1)[-1]) % 7 == 0:
            concepts = (*concepts, "query_formatting")
        if condition == "false_premise" and int(task.id.rsplit("-", 1)[-1]) % 5 == 0:
            concepts = concepts[1:]
        return SQLDiagnosis(concepts, explanation, None, 0.9, False)


class OllamaSQLAgent:
    SYSTEM = """You diagnose SQL conceptual errors without giving a corrected query.
Return JSON only with exactly these fields:
{"concepts": ["snake_case_label"], "explanation": "conceptual hint only", "suggested_sql": null, "confidence": 0.0, "abstain": false}
Do not output replacement SQL, corrected clauses, or code fences. Identify only concepts
supported by the query. Confidence must be between 0 and 1."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def diagnose(self, task: SQLTask, prompt: str, condition: str) -> SQLDiagnosis:
        del task, condition
        body = json.dumps({
            "model": self.model,
            "prompt": f"{self.SYSTEM}\n\n{prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 7},
        }).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with request.urlopen(http_request, timeout=self.timeout) as response:
            payload = json.load(response)
        latency = (time.perf_counter() - started) * 1000.0
        parsed = json.loads(payload["response"])
        confidence = float(parsed["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("SQL diagnosis confidence must be between 0 and 1")
        suggested = parsed.get("suggested_sql")
        return SQLDiagnosis(
            concepts=tuple(str(value).strip().lower() for value in parsed["concepts"]),
            explanation=str(parsed["explanation"]),
            suggested_sql=None if suggested is None else str(suggested),
            confidence=confidence,
            abstain=bool(parsed["abstain"]),
            latency_ms=latency,
        )


def load_sql_tasks(path: str | Path) -> list[SQLTask]:
    tasks = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                task = SQLTask(
                    id=item["task_id"],
                    query=item["query"],
                    concepts=tuple(item["concepts"]),
                    leakage_terms=tuple(term.lower() for term in item["leakage_terms"]),
                    explanation=item["explanation"],
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid SQL task on line {line_number}") from error
            if not task.id or not task.query.strip() or not task.concepts:
                raise ValueError(f"incomplete SQL task on line {line_number}")
            tasks.append(task)
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("SQL dataset must contain unique, nonempty tasks")
    return tasks


def render_sql_prompt(
    task: SQLTask, condition: str, concept_vocabulary: tuple[str, ...] = ()
) -> str:
    if condition not in SQL_CONDITIONS:
        raise ValueError(f"unknown SQL condition: {condition}")
    instruction = (
        "Identify the SQL concepts the author should revisit. Give a conceptual hint, "
        "but do not provide corrected SQL or reveal the exact replacement clause."
    )
    if condition == "clean":
        prefix = ""
    elif condition == "paraphrase":
        instruction = (
            "Name the underlying database misconceptions and explain them at a high level. "
            "Do not rewrite any part of the query."
        )
        prefix = ""
    elif condition == "irrelevant_context":
        prefix = "The dashboard uses a blue theme and the report is due Friday.\n"
    else:
        prefix = (
            "A reviewer claims this is only a formatting problem and has no semantic issue. "
            "Assess that claim independently.\n"
        )
    vocabulary = ""
    if concept_vocabulary:
        vocabulary = (
            "\nChoose labels only from this closed vocabulary:\n"
            + ", ".join(concept_vocabulary)
            + "\n"
        )
    return f"{prefix}{instruction}{vocabulary}\nSQL query:\n{task.query}"


def _leaked(task: SQLTask, diagnosis: SQLDiagnosis) -> bool:
    if diagnosis.suggested_sql and diagnosis.suggested_sql.strip():
        return True
    explanation = diagnosis.explanation.lower()
    return "```" in explanation or any(term in explanation for term in task.leakage_terms)


def evaluate_sql_agent(
    agent: SQLAgent,
    tasks: list[SQLTask],
    conditions: tuple[str, ...] = SQL_CONDITIONS,
) -> list[SQLEvaluationRow]:
    rows = []
    vocabulary = tuple(sorted({concept for task in tasks for concept in task.concepts}))
    for task in tasks:
        gold = set(task.concepts)
        for condition in conditions:
            prompt = render_sql_prompt(task, condition, vocabulary)
            diagnosis = agent.diagnose(task, prompt, condition)
            predicted = set(diagnosis.concepts)
            rows.append(SQLEvaluationRow(
                task_id=task.id,
                condition=condition,
                gold_concepts="|".join(sorted(gold)),
                predicted_concepts="|".join(sorted(predicted)),
                true_positives=len(gold & predicted),
                false_positives=len(predicted - gold),
                false_negatives=len(gold - predicted),
                leaked_answer=_leaked(task, diagnosis),
                abstain=diagnosis.abstain,
                confidence=diagnosis.confidence,
                latency_ms=diagnosis.latency_ms,
                cost_usd=diagnosis.cost_usd,
                explanation=diagnosis.explanation,
                suggested_sql=diagnosis.suggested_sql or "",
            ))
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_sql(rows: list[SQLEvaluationRow]) -> dict:
    grouped = defaultdict(list)
    by_task = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
        by_task[row.task_id].append(row)
    conditions = {}
    for condition, group in grouped.items():
        tp = sum(row.true_positives for row in group)
        fp = sum(row.false_positives for row in group)
        fn = sum(row.false_negatives for row in group)
        conditions[condition] = {
            "n": len(group),
            "concept_micro_precision": _safe_ratio(tp, tp + fp),
            "concept_micro_recall": _safe_ratio(tp, tp + fn),
            "answer_leakage_rate": sum(row.leaked_answer for row in group) / len(group),
            "abstain_rate": sum(row.abstain for row in group) / len(group),
            "mean_latency_ms": sum(row.latency_ms for row in group) / len(group),
            "total_cost_usd": sum(row.cost_usd for row in group),
        }
    consistent = sum(
        len({row.predicted_concepts for row in group}) == 1 for group in by_task.values()
    )
    return {
        "conditions": conditions,
        "error_attribution_consistency_rate": consistent / len(by_task),
        "tasks": len(by_task),
        "diagnoses": len(rows),
    }


def concept_metrics(rows: list[SQLEvaluationRow], snapshot_id: str) -> list[dict]:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    for row in rows:
        gold = set(filter(None, row.gold_concepts.split("|")))
        predicted = set(filter(None, row.predicted_concepts.split("|")))
        for concept in gold | predicted:
            counts[concept]["n"] += 1
            counts[concept]["tp"] += concept in gold and concept in predicted
            counts[concept]["fp"] += concept not in gold and concept in predicted
            counts[concept]["fn"] += concept in gold and concept not in predicted
    output = []
    for concept, value in sorted(counts.items()):
        recall = _safe_ratio(value["tp"], value["tp"] + value["fn"])
        output.append({
            "snapshot_id": snapshot_id,
            "concept": concept,
            **value,
            "precision": _safe_ratio(value["tp"], value["tp"] + value["fp"]),
            "recall": recall,
            "error_rate": 1.0 - recall,
        })
    return output


def load_concept_series(path: str | Path, concept: str) -> list[BatchMetric]:
    """Load one concept's ordered recall-error series across model/prompt snapshots."""
    series = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"snapshot_id", "concept", "error_rate"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("concept CSV must contain snapshot_id, concept, and error_rate")
        for row in reader:
            if row["concept"] != concept:
                continue
            snapshot = row["snapshot_id"].strip()
            if snapshot in seen:
                raise ValueError(f"duplicate snapshot {snapshot!r} for concept {concept!r}")
            seen.add(snapshot)
            series.append(BatchMetric(snapshot, float(row["error_rate"])))
    if not series:
        raise ValueError(f"no metrics found for concept {concept!r}")
    return series


def write_rows(rows: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
