import argparse
from dataclasses import asdict
import json

from .data import load_jsonl
from .evaluation import evaluate, summarize, write_csv
from .models import KeywordBaseline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="shiftwatch")
    parser.add_argument("dataset", help="JSONL file with id, text, and label")
    parser.add_argument("--output", default="results/evaluation.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    examples = load_jsonl(args.dataset)
    rows = evaluate(KeywordBaseline(), examples, seed=args.seed)
    write_csv(rows, args.output)
    print(json.dumps({"summary": summarize(rows), "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
