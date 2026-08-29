from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

from .llm import FixtureLLM, LLM


@dataclass(frozen=True)
class LLMCase:
    id: str
    required_terms: tuple[str, ...]
    prompts: dict[str, str]
    refutation_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMEvaluationRow:
    case_id: str
    condition: str
    prompt: str
    answer: str
    confidence: float
    abstain: bool
    correct: bool
    refuted_false_premise: bool
    confidently_wrong: bool


def load_llm_cases(path: str | Path) -> tuple[list[LLMCase], dict[str, dict]]:
    cases = []
    responses = {}
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                prompts = dict(item["prompts"])
                cases.append(LLMCase(
                    id=item["id"],
                    required_terms=tuple(term.lower() for term in item["required_terms"]),
                    prompts=prompts,
                    refutation_terms=tuple(
                        term.lower() for term in item.get("refutation_terms", [])
                    ),
                ))
                for condition, payload in item.get("fixture_responses", {}).items():
                    responses[prompts[condition]] = payload
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid LLM case on line {line_number}") from error
    if not cases:
        raise ValueError("LLM dataset must contain at least one case")
    return cases, responses


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def evaluate_llm(
    model: LLM,
    cases: list[LLMCase],
    confidence_threshold: float = 0.8,
) -> list[LLMEvaluationRow]:
    rows = []
    for case in cases:
        for condition, prompt in case.prompts.items():
            response = model.generate(prompt)
            correct = not response.abstain and _contains_any(response.answer, case.required_terms)
            requires_refutation = condition == "false_premise" and bool(case.refutation_terms)
            refuted = requires_refutation and _contains_any(
                response.answer, case.refutation_terms
            )
            rows.append(LLMEvaluationRow(
                case_id=case.id,
                condition=condition,
                prompt=prompt,
                answer=response.answer,
                confidence=response.confidence,
                abstain=response.abstain,
                correct=correct,
                refuted_false_premise=refuted,
                confidently_wrong=not correct and response.confidence >= confidence_threshold,
            ))
    return rows


def summarize_llm(rows: list[LLMEvaluationRow]) -> dict:
    grouped = defaultdict(list)
    by_case = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
        by_case[row.case_id].append(row)

    conditions = {}
    for condition, group in grouped.items():
        n = len(group)
        conditions[condition] = {
            "n": n,
            "accuracy": sum(row.correct for row in group) / n,
            "abstain_rate": sum(row.abstain for row in group) / n,
            "mean_confidence": sum(row.confidence for row in group) / n,
            "confidently_wrong_rate": sum(row.confidently_wrong for row in group) / n,
        }
        if condition == "false_premise":
            conditions[condition]["refutation_rate"] = sum(
                row.refuted_false_premise for row in group
            ) / n

    consistent_cases = sum(
        len({row.correct for row in group}) == 1 for group in by_case.values()
    )
    return {
        "conditions": conditions,
        "behavioral_consistency_rate": consistent_cases / len(by_case),
        "cases": len(by_case),
    }


def write_llm_csv(rows: list[LLMEvaluationRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def fixture_model(path: str | Path) -> tuple[list[LLMCase], FixtureLLM]:
    cases, responses = load_llm_cases(path)
    return cases, FixtureLLM(responses)
