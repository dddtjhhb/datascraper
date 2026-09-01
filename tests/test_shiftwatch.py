import tempfile
import unittest
from pathlib import Path

from shiftwatch.data import load_jsonl
from shiftwatch.code_evaluation import (
    FixtureCodeAgent,
    CodeCandidate,
    CodeTask,
    evaluate_code_agent,
    execute_candidate,
    load_code_tasks,
    render_code_prompt,
    summarize_code,
)
from shiftwatch.evaluation import evaluate, summarize, write_csv
from shiftwatch.models import KeywordBaseline
from shiftwatch.llm import parse_structured_response
from shiftwatch.llm import FixtureLLM
from shiftwatch.llm_evaluation import (
    LLMCase,
    evaluate_llm,
    fixture_model,
    load_llm_cases,
    summarize_llm,
)
from shiftwatch.monitoring import cusum, ewma, load_batch_metrics, monitor, write_alarms
from shiftwatch.perturbations import apply


ROOT = Path(__file__).parents[1]


class EvaluationTest(unittest.TestCase):
    def test_evaluation_is_reproducible(self):
        examples = load_jsonl(ROOT / "datasets/demo.jsonl")
        first = evaluate(KeywordBaseline(), examples, seed=11)
        second = evaluate(KeywordBaseline(), examples, seed=11)
        self.assertEqual(first, second)
        summary = summarize(first)
        self.assertEqual(set(summary), {"clean", "typo", "truncate", "distractor", "injection"})
        self.assertEqual(summary["clean"]["n"], 10)

    def test_injection_does_not_change_gold_label(self):
        text = apply("A scientist published a study.", "injection", 1)
        self.assertIn("Ignore the classification task", text)

    def test_csv_export(self):
        rows = evaluate(KeywordBaseline(), load_jsonl(ROOT / "datasets/demo.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.csv"
            write_csv(rows, output)
            self.assertEqual(len(output.read_text().splitlines()), len(rows) + 1)


class MonitoringTest(unittest.TestCase):
    def test_detectors_alarm_after_error_increase(self):
        values = [0.05] * 8 + [0.40] * 5
        self.assertTrue(cusum(values, target=0.05))
        self.assertTrue(ewma(values, target=0.05))

    def test_batch_series_connects_to_named_alarms(self):
        metrics = load_batch_metrics(ROOT / "datasets/demo_batch_metrics.csv")
        alarms = monitor(metrics, target=0.05)
        self.assertTrue(alarms)
        self.assertIn("batch_id", alarms[0])
        self.assertGreaterEqual(alarms[0]["index"], 8)

    def test_alarm_export_has_header_even_without_alarms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "alarms.csv"
            write_alarms([], output)
            self.assertEqual(
                output.read_text().strip(),
                "batch_id,index,method,statistic,error_rate",
            )


class LLMEvaluationTest(unittest.TestCase):
    def test_extended_benchmark_has_60_cases_and_four_conditions(self):
        cases, _ = load_llm_cases(ROOT / "datasets/llm_benchmark_60.jsonl")
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({case.id for case in cases}), 60)
        self.assertEqual(
            {condition for case in cases for condition in case.prompts},
            {"clean", "paraphrase", "distractor", "false_premise"},
        )
        self.assertEqual(sum(case.should_abstain for case in cases), 10)

    def test_structured_response_parser(self):
        response = parse_structured_response(
            '{"answer":"Mercury","confidence":0.9,"abstain":false}'
        )
        self.assertEqual(response.answer, "Mercury")
        self.assertEqual(response.confidence, 0.9)

    def test_fixture_evaluates_four_behavior_conditions(self):
        cases, model = fixture_model(ROOT / "datasets/llm_demo.jsonl")
        rows = evaluate_llm(model, cases)
        self.assertEqual(len(rows), 12)
        summary = summarize_llm(rows)
        self.assertEqual(
            set(summary["conditions"]),
            {"clean", "paraphrase", "distractor", "false_premise"},
        )
        self.assertAlmostEqual(
            summary["conditions"]["false_premise"]["refutation_rate"], 2 / 3
        )
        self.assertAlmostEqual(summary["behavioral_consistency_rate"], 2 / 3)
        self.assertAlmostEqual(
            summary["conditions"]["false_premise"]["confidently_wrong_rate"], 1 / 3
        )

    def test_rejection_without_correction_is_refutation_but_not_correct(self):
        case = LLMCase(
            id="example",
            category="test",
            required_terms=("correct fact",),
            prompts={"false_premise": "An incorrect claim, right?"},
            refutation_terms=("incorrect",),
        )
        model = FixtureLLM({
            "An incorrect claim, right?": {
                "answer": "No",
                "confidence": 0.9,
                "abstain": False,
            }
        })
        row = evaluate_llm(model, [case])[0]
        self.assertTrue(row.refuted_false_premise)
        self.assertFalse(row.correct)

    def test_semantic_abstention_is_detected_when_flag_is_inconsistent(self):
        case = LLMCase(
            id="unknown",
            category="uncertainty",
            required_terms=(),
            prompts={"clean": "What will a fair coin show tomorrow?"},
            should_abstain=True,
        )
        model = FixtureLLM({
            "What will a fair coin show tomorrow?": {
                "answer": "It is impossible to predict with certainty.",
                "confidence": 0.9,
                "abstain": False,
            }
        })
        row = evaluate_llm(model, [case])[0]
        self.assertTrue(row.semantic_abstention)
        self.assertTrue(row.correct)


class CodeAgentEvaluationTest(unittest.TestCase):
    def test_fixture_runs_repeated_controlled_conditions(self):
        tasks = load_code_tasks(ROOT / "datasets/code_tasks_demo.jsonl")
        rows = evaluate_code_agent(FixtureCodeAgent(), tasks, repeats=3)
        self.assertEqual(len(rows), 4 * 4 * 3)
        summary = summarize_code(rows)
        self.assertEqual(summary["tasks"], 4)
        self.assertEqual(summary["runs"], 48)
        self.assertEqual(summary["conditions"]["clean"]["pass_rate"], 1.0)
        self.assertLess(summary["conditions"]["false_premise"]["pass_rate"], 1.0)

    def test_prompt_conditions_are_distinct(self):
        task = load_code_tasks(ROOT / "datasets/code_tasks_demo.jsonl")[0]
        prompts = {
            condition: render_code_prompt(task, condition)
            for condition in ("clean", "irrelevant_context", "false_premise", "long_context")
        }
        self.assertEqual(len(set(prompts.values())), 4)
        self.assertGreater(len(prompts["long_context"]), len(prompts["clean"]) * 10)

    def test_executor_rejects_imports_before_running(self):
        task = CodeTask(
            id="unsafe",
            prompt="unsafe",
            entry_point="solve",
            tests=("solve() == 1",),
            false_premise="none",
            fixture_candidates={},
        )
        result = execute_candidate(
            task, CodeCandidate("import os\ndef solve():\n    return 1")
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "policy_rejection")

    def test_executor_classifies_assertion_failure(self):
        task = CodeTask(
            id="wrong",
            prompt="wrong",
            entry_point="solve",
            tests=("solve() == 2",),
            false_premise="none",
            fixture_candidates={},
        )
        result = execute_candidate(task, CodeCandidate("def solve():\n    return 1"))
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "assertion_failure")


if __name__ == "__main__":
    unittest.main()
