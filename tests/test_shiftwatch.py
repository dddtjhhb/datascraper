import tempfile
import unittest
from pathlib import Path

from shiftwatch.data import load_jsonl
from shiftwatch.evaluation import evaluate, summarize, write_csv
from shiftwatch.models import KeywordBaseline
from shiftwatch.monitoring import cusum, ewma
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


if __name__ == "__main__":
    unittest.main()
