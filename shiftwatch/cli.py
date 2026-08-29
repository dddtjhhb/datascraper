import argparse
import json

from .data import load_jsonl
from .evaluation import evaluate, summarize, write_csv
from .models import KeywordBaseline
from .monitoring import load_batch_metrics, monitor, write_alarms
from .llm import OllamaLLM
from .llm_evaluation import (
    evaluate_llm,
    fixture_model,
    load_llm_cases,
    recorded_model,
    summarize_llm,
    write_llm_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shiftwatch")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a model under controlled shifts")
    evaluate_parser.add_argument("dataset", help="JSONL file with id, text, and label")
    evaluate_parser.add_argument("--output", default="results/evaluation.csv")
    evaluate_parser.add_argument("--seed", type=int, default=7)

    llm_parser = commands.add_parser(
        "llm-evaluate", help="evaluate LLM behavior across prompt variants"
    )
    llm_parser.add_argument("dataset", help="JSONL file containing LLM evaluation cases")
    llm_parser.add_argument(
        "--backend", choices=("fixture", "ollama", "recorded"), default="fixture"
    )
    llm_parser.add_argument("--model", help="Ollama model name, for example llama3.2:3b")
    llm_parser.add_argument("--responses", help="prior result CSV for recorded backend")
    llm_parser.add_argument("--base-url", default="http://localhost:11434")
    llm_parser.add_argument("--response-mode", choices=("short", "free"), default="short")
    llm_parser.add_argument("--output", default="results/llm_evaluation.csv")

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
    elif args.command == "llm-evaluate":
        if args.backend == "fixture":
            cases, model = fixture_model(args.dataset)
        elif args.backend == "ollama":
            if not args.model:
                raise SystemExit("--model is required with --backend ollama")
            cases, _ = load_llm_cases(args.dataset)
            model = OllamaLLM(
                args.model,
                base_url=args.base_url,
                response_mode=args.response_mode,
            )
        else:
            if not args.responses:
                raise SystemExit("--responses is required with --backend recorded")
            cases, _ = load_llm_cases(args.dataset)
            model = recorded_model(args.responses)
        rows = evaluate_llm(model, cases)
        write_llm_csv(rows, args.output)
        result = {"summary": summarize_llm(rows), "output": args.output}
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
