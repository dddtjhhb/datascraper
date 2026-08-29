import argparse
import json

from .data import load_jsonl
from .evaluation import evaluate, summarize, write_csv
from .models import KeywordBaseline
from .monitoring import load_batch_metrics, monitor, write_alarms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shiftwatch")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a model under controlled shifts")
    evaluate_parser.add_argument("dataset", help="JSONL file with id, text, and label")
    evaluate_parser.add_argument("--output", default="results/evaluation.csv")
    evaluate_parser.add_argument("--seed", type=int, default=7)

    monitor_parser = commands.add_parser("monitor", help="monitor an ordered batch error-rate series")
    monitor_parser.add_argument("metrics", help="CSV file with batch_id and error_rate")
    monitor_parser.add_argument("--target", type=float, required=True, help="expected baseline error rate")
    monitor_parser.add_argument("--output", default="results/alarms.csv")
    monitor_parser.add_argument("--cusum-drift", type=float, default=0.02)
    monitor_parser.add_argument("--cusum-threshold", type=float, default=0.30)
    monitor_parser.add_argument("--ewma-alpha", type=float, default=0.30)
    monitor_parser.add_argument("--ewma-threshold", type=float, default=0.15)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        examples = load_jsonl(args.dataset)
        rows = evaluate(KeywordBaseline(), examples, seed=args.seed)
        write_csv(rows, args.output)
        result = {"summary": summarize(rows), "output": args.output}
    else:
        metrics = load_batch_metrics(args.metrics)
        alarms = monitor(
            metrics,
            target=args.target,
            cusum_drift=args.cusum_drift,
            cusum_threshold=args.cusum_threshold,
            ewma_alpha=args.ewma_alpha,
            ewma_threshold=args.ewma_threshold,
        )
        write_alarms(alarms, args.output)
        result = {
            "batches": len(metrics),
            "target": args.target,
            "alarms": alarms,
            "output": args.output,
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
