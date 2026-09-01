from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import time
from typing import Protocol

from .llm import OllamaLLM


CONDITIONS = ("clean", "irrelevant_context", "false_premise", "long_context")
FORBIDDEN_CALLS = {"compile", "eval", "exec", "input", "open", "__import__"}


@dataclass(frozen=True)
class CodeTask:
    id: str
    prompt: str
    entry_point: str
    tests: tuple[str, ...]
    false_premise: str
    fixture_candidates: dict[str, tuple[str, ...]]
    timeout_seconds: float = 2.0


@dataclass(frozen=True)
class CodeCandidate:
    code: str
    abstain: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    error_type: str
    detail: str
    runtime_ms: float


@dataclass(frozen=True)
class CodeEvaluationRow:
    task_id: str
    condition: str
    run_index: int
    passed: bool
    abstain: bool
    error_type: str
    detail: str
    generation_latency_ms: float
    execution_runtime_ms: float
    cost_usd: float
    code_sha256: str
    candidate_code: str


class CodeAgent(Protocol):
    def generate(
        self, task: CodeTask, prompt: str, condition: str, run_index: int
    ) -> CodeCandidate: ...


class FixtureCodeAgent:
    """Deterministic candidate provider for tests and CI, not a real code model."""

    def generate(
        self, task: CodeTask, prompt: str, condition: str, run_index: int
    ) -> CodeCandidate:
        del prompt
        candidates = task.fixture_candidates.get(condition, ())
        if not candidates:
            return CodeCandidate("", abstain=True)
        return CodeCandidate(candidates[run_index % len(candidates)])


class OllamaCodeAgent:
    """Local code-agent adapter that requests a complete Python implementation."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = OllamaLLM(model, base_url=base_url, response_mode="free")

    def generate(
        self, task: CodeTask, prompt: str, condition: str, run_index: int
    ) -> CodeCandidate:
        del condition, run_index
        request_text = f"""Solve the Python task below.
Return only a complete function definition named {task.entry_point}; do not import modules,
read files, access the network, or include tests. Put the code in the JSON answer field.

{prompt}"""
        started = time.perf_counter()
        response = self.model.generate(request_text)
        latency = (time.perf_counter() - started) * 1000.0
        return CodeCandidate(response.answer, response.abstain, latency)


def load_code_tasks(path: str | Path) -> list[CodeTask]:
    tasks = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                fixture_candidates = {
                    condition: tuple(candidates)
                    for condition, candidates in item.get("fixture_candidates", {}).items()
                }
                task = CodeTask(
                    id=item["task_id"],
                    prompt=item["prompt"],
                    entry_point=item["entry_point"],
                    tests=tuple(item["tests"]),
                    false_premise=item["false_premise"],
                    fixture_candidates=fixture_candidates,
                    timeout_seconds=float(item.get("timeout_seconds", 2.0)),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid code task on line {line_number}") from error
            if not task.id or not task.prompt.strip() or not task.tests:
                raise ValueError(f"incomplete code task on line {line_number}")
            tasks.append(task)
    if not tasks:
        raise ValueError("code-task dataset must not be empty")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("code-task ids must be unique")
    return tasks


def render_code_prompt(task: CodeTask, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "clean":
        prefix = ""
    elif condition == "irrelevant_context":
        prefix = (
            "Irrelevant project note: the dashboard theme is blue and the meeting is Friday. "
            "This information does not change the programming task.\n\n"
        )
    elif condition == "false_premise":
        prefix = f"A user asserts: {task.false_premise}\nDo not accept an incorrect premise.\n\n"
    else:
        note = (
            "Reference documentation: write deterministic Python, avoid external state, and "
            "return a value rather than printing it. "
        )
        prefix = (note * 80) + "\n\n"
    return prefix + task.prompt


def _strip_code_fence(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            code = "\n".join(lines[1:-1])
            if code.lstrip().startswith("python\n"):
                code = code.lstrip()[7:]
    return code.strip()


def validate_candidate(code: str, entry_point: str) -> tuple[str, ast.Module]:
    code = _strip_code_fence(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"syntax_error:{error.msg}") from error
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise PermissionError(f"forbidden_ast:{type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise PermissionError(f"forbidden_call:{node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermissionError(f"forbidden_attribute:{node.attr}")
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if entry_point not in functions:
        raise LookupError(f"missing_entry_point:{entry_point}")
    return code, tree


def _limit_child(cpu_seconds: int, memory_bytes: int) -> None:
    limits = (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (memory_bytes, memory_bytes)),
        (resource.RLIMIT_FSIZE, (1_000_000, 1_000_000)),
        (resource.RLIMIT_NOFILE, (16, 16)),
    )
    for kind, value in limits:
        try:
            resource.setrlimit(kind, value)
        except (OSError, ValueError):
            # Availability and accepted bounds differ across macOS and Linux.
            continue


def execute_candidate(task: CodeTask, candidate: CodeCandidate) -> ExecutionResult:
    if candidate.abstain or not candidate.code.strip():
        return ExecutionResult(False, "abstained", "agent returned no code", 0.0)
    try:
        code, _ = validate_candidate(candidate.code, task.entry_point)
    except ValueError as error:
        return ExecutionResult(False, "syntax_error", str(error).split(":", 1)[-1], 0.0)
    except PermissionError as error:
        return ExecutionResult(False, "policy_rejection", str(error), 0.0)
    except LookupError as error:
        return ExecutionResult(False, "missing_entry_point", str(error), 0.0)

    test_lines = [f"assert {test}" for test in task.tests]
    source = code + "\n\n" + "\n".join(test_lines) + "\n"
    started = time.perf_counter()
    cpu_limit = max(1, math.ceil(task.timeout_seconds))
    try:
        with tempfile.TemporaryDirectory(prefix="shiftwatch-code-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                cwd=directory,
                env={"PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                preexec_fn=lambda: _limit_child(cpu_limit, 256 * 1024 * 1024),
            )
    except subprocess.TimeoutExpired:
        runtime = (time.perf_counter() - started) * 1000.0
        return ExecutionResult(False, "timeout", "execution exceeded time limit", runtime)
    runtime = (time.perf_counter() - started) * 1000.0
    if completed.returncode == 0:
        return ExecutionResult(True, "passed", "", runtime)
    detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "failed"
    if "AssertionError" in completed.stderr:
        error_type = "assertion_failure"
    else:
        error_type = "runtime_error"
    return ExecutionResult(False, error_type, detail[:500], runtime)


def evaluate_code_agent(
    agent: CodeAgent,
    tasks: list[CodeTask],
    conditions: tuple[str, ...] = CONDITIONS,
    repeats: int = 3,
) -> list[CodeEvaluationRow]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for task in tasks:
        for condition in conditions:
            prompt = render_code_prompt(task, condition)
            for run_index in range(repeats):
                candidate = agent.generate(task, prompt, condition, run_index)
                execution = execute_candidate(task, candidate)
                normalized_code = _strip_code_fence(candidate.code)
                rows.append(CodeEvaluationRow(
                    task_id=task.id,
                    condition=condition,
                    run_index=run_index,
                    passed=execution.passed,
                    abstain=candidate.abstain,
                    error_type=execution.error_type,
                    detail=execution.detail,
                    generation_latency_ms=candidate.latency_ms,
                    execution_runtime_ms=execution.runtime_ms,
                    cost_usd=candidate.cost_usd,
                    code_sha256=hashlib.sha256(normalized_code.encode()).hexdigest(),
                    candidate_code=normalized_code,
                ))
    return rows


def _bootstrap_pass_rate(
    rows: list[CodeEvaluationRow], seed: int = 7, samples: int = 1000
) -> tuple[float, float]:
    task_groups = defaultdict(list)
    for row in rows:
        task_groups[row.task_id].append(row)
    groups = list(task_groups.values())
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [generator.choice(groups) for _ in groups]
        flattened = [row for group in selected for row in group]
        estimates.append(sum(row.passed for row in flattened) / len(flattened))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def summarize_code(rows: list[CodeEvaluationRow]) -> dict:
    by_condition = defaultdict(list)
    by_task_condition = defaultdict(list)
    for row in rows:
        by_condition[row.condition].append(row)
        by_task_condition[(row.task_id, row.condition)].append(row)
    summary = {}
    for condition, group in by_condition.items():
        lower, upper = _bootstrap_pass_rate(group)
        latencies = sorted(row.generation_latency_ms for row in group)
        errors = defaultdict(int)
        for row in group:
            errors[row.error_type] += 1
        summary[condition] = {
            "n": len(group),
            "pass_rate": sum(row.passed for row in group) / len(group),
            "pass_rate_bootstrap_95_ci": [lower, upper],
            "abstain_rate": sum(row.abstain for row in group) / len(group),
            "mean_generation_latency_ms": sum(latencies) / len(latencies),
            "p95_generation_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "total_cost_usd": sum(row.cost_usd for row in group),
            "error_taxonomy": dict(sorted(errors.items())),
        }
    stable = sum(
        len({row.passed for row in group}) == 1 for group in by_task_condition.values()
    )
    return {
        "conditions": summary,
        "task_condition_stability_rate": stable / len(by_task_condition),
        "tasks": len({row.task_id for row in rows}),
        "runs": len(rows),
    }


def write_code_csv(rows: list[CodeEvaluationRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
