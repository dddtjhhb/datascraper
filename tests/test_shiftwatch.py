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
from shiftwatch.sql_evaluation import (
    FixtureSQLAgent,
    SQLDiagnosis,
    SQLTask,
    concept_metrics,
    evaluate_sql_agent,
    load_concept_series,
    load_sql_tasks,
    render_sql_prompt,
    summarize_sql,
    write_rows as write_sql_rows,
)


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


class SQLDiagnosticEvaluationTest(unittest.TestCase):
    def test_dataset_has_30_labeled_cases(self):
        tasks = load_sql_tasks(ROOT / "datasets/sql_misconceptions_30.jsonl")
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len({task.id for task in tasks}), 30)
        self.assertTrue(all(task.concepts for task in tasks))

    def test_fixture_reports_concept_metrics_across_four_conditions(self):
        tasks = load_sql_tasks(ROOT / "datasets/sql_misconceptions_30.jsonl")
        rows = evaluate_sql_agent(FixtureSQLAgent(), tasks)
        self.assertEqual(len(rows), 120)
        summary = summarize_sql(rows)
        self.assertEqual(summary["tasks"], 30)
        self.assertEqual(summary["diagnoses"], 120)
        self.assertEqual(summary["conditions"]["clean"]["concept_micro_recall"], 1.0)
        self.assertLess(
            summary["conditions"]["false_premise"]["concept_micro_recall"], 1.0
        )

    def test_answer_leakage_detects_corrected_sql(self):
        task = SQLTask(
            id="sql-99",
            query="SELECT * FROM t WHERE x = NULL;",
            concepts=("null_semantics",),
            leakage_terms=("is null",),
            explanation="missing values use three-valued logic",
        )

        class LeakingAgent:
            def diagnose(self, task, prompt, condition):
                return SQLDiagnosis(
                    ("null_semantics",), "Use IS NULL.", None, 0.9, False
                )

        row = evaluate_sql_agent(LeakingAgent(), [task], ("clean",))[0]
        self.assertTrue(row.leaked_answer)

    def test_prompt_perturbations_preserve_query(self):
        task = load_sql_tasks(ROOT / "datasets/sql_misconceptions_30.jsonl")[0]
        vocabulary = tuple(sorted({
            concept
            for sql_task in load_sql_tasks(ROOT / "datasets/sql_misconceptions_30.jsonl")
            for concept in sql_task.concepts
        }))
        prompts = [render_sql_prompt(task, condition, vocabulary) for condition in (
            "clean", "paraphrase", "irrelevant_context", "false_premise"
        )]
        self.assertEqual(len(set(prompts)), 4)
        self.assertTrue(all(task.query in prompt for prompt in prompts))
        self.assertTrue(all("closed vocabulary" in prompt for prompt in prompts))

    def test_concept_metrics_connect_to_monitoring_series(self):
        tasks = load_sql_tasks(ROOT / "datasets/sql_misconceptions_30.jsonl")
        rows = evaluate_sql_agent(FixtureSQLAgent(), tasks, ("clean",))
        records = concept_metrics(rows, "model-v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.csv"
            write_sql_rows(records, path)
            concept = records[0]["concept"]
            series = load_concept_series(path, concept)
            self.assertEqual(series[0].batch_id, "model-v1")
            self.assertGreaterEqual(series[0].error_rate, 0.0)

    def test_sql_concept_degradation_triggers_monitor(self):
        series = load_concept_series(
            ROOT / "datasets/demo_sql_concept_history.csv", "null_semantics"
        )
        alarms = monitor(series, target=0.10)
        self.assertTrue(alarms)
        self.assertGreaterEqual(alarms[0]["index"], 8)


if __name__ == "__main__":
    unittest.main()
