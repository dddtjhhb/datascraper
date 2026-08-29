# ShiftWatch

ShiftWatch is a reproducible evaluation pipeline for studying how language-model reliability changes under controlled input distribution shifts. Its main direction is LLM behavioral evaluation; a transparent text-classification baseline remains as a pipeline validation case study.

This project is motivated by trustworthy ML and AI safety evaluation. It does **not** claim to solve AI alignment. Its narrower goal is to make model failures measurable, reproducible, and visible before deploying more complex training methods.

## Research questions

- How much does model accuracy degrade under typos, truncation, irrelevant context, and prompt injection?
- Does a model abstain when evidence is insufficient, or remain confidently wrong?
- Are reported confidence scores calibrated under shift?
- How quickly can CUSUM or EWMA detect an increase in the model's failure rate?
- Later: does parameter-efficient fine-tuning improve average accuracy at the cost of out-of-distribution reliability?

## Current scope: Phase 1

The first phase is intentionally CPU-only and dependency-light:

- validated JSONL dataset contract;
- deterministic perturbation generation;
- transparent keyword baseline;
- accuracy, error, abstention, confidence, and Brier-score metrics;
- clean-versus-shift comparison;
- CUSUM and EWMA degradation detectors;
- CSV output for failure analysis;
- automated tests and deterministic seeds.

The keyword model is not presented as a competitive classifier. It is a transparent baseline that verifies the evaluation system before an LLM is added.

## LLM behavioral evaluation

The LLM path asks the same factual question under four conditions: clean wording, a paraphrase, irrelevant context, and a user-supplied false premise. Each backend returns a structured answer, confidence score, and abstention decision. ShiftWatch measures correctness, behavioral consistency, false-premise refutation, and confidently wrong answers.

The included `fixture` backend contains recorded responses only so tests and GitHub Actions stay deterministic. It is explicitly not a real model result. A real local model can be served through Ollama:

```bash
python -m shiftwatch.cli llm-evaluate datasets/llm_demo.jsonl \
  --backend ollama --model llama3.2:3b \
  --output results/llm_evaluation.csv
```

For a larger experiment, the versioned starter benchmark contains 60 cases and four prompt conditions per case (240 generations). Free-response mode keeps the model's full explanatory paragraph inside the structured result so it can be reviewed as well as scored:

```bash
python scripts/build_llm_benchmark.py
python -m shiftwatch.cli llm-evaluate datasets/llm_benchmark_60.jsonl \
  --backend ollama --model llama3:latest --response-mode free \
  --output results/llama3_benchmark_60_free.csv
```

The benchmark spans factual knowledge, reasoning, computing, statistics, monitoring, ML, AI safety concepts, and ten intentionally unknowable questions. Metrics are broken down by prompt condition and category. The CSV preserves every full answer and records answer length, mentions of forbidden terms, explicit versus semantic abstention, false-premise refutation, and high-confidence errors. Mentioning an incorrect term is an audit signal rather than an automatic failure because a good correction may quote the false claim. The starter cases are suitable for pipeline development, not a publishable benchmark; facts and rubrics require independent review before making research claims.

Previously generated responses can be rescored after improving a rubric without paying the inference cost again:

```bash
python -m shiftwatch.cli llm-evaluate datasets/llm_benchmark_60.jsonl \
  --backend recorded --responses results/llama3_benchmark_60_free.csv \
  --output results/llama3_benchmark_60_rescored.csv
```

The same adapter boundary can later point to an Ollama server running on a university GPU node. Cluster access, job scheduling, and model storage depend on the account and allocation supplied by the university, course, or research group.

## Architecture

```text
versioned dataset
      |
controlled perturbations
      |
model adapter -> structured Prediction
      |
reliability metrics + failure table
      |
batch error-rate time series
      |
CUSUM / EWMA degradation alarms
```

## Quick start

Python 3.10+ is sufficient.

```bash
python -m shiftwatch.cli evaluate datasets/demo.jsonl --output results/evaluation.csv
python -m shiftwatch.cli llm-evaluate datasets/llm_demo.jsonl \
  --backend fixture --output results/llm_evaluation.csv
python -m shiftwatch.cli monitor datasets/demo_batch_metrics.csv \
  --target 0.05 --output results/alarms.csv
python -m unittest discover -v
```

The CLI prints a JSON summary and writes one row per example and condition. Fixed random seeds make perturbations reproducible.

The monitoring command expects an ordered CSV series:

```csv
batch_id,error_rate
2026-08-01,0.05
2026-08-02,0.06
```

Each batch might represent a day, deployment window, or fixed number of predictions. ShiftWatch does not treat perturbation conditions as time points: batch error rates must come from genuinely ordered evaluation or production windows. The target is supplied explicitly so the monitoring baseline is documented rather than estimated from the same period being tested.

## Run without a local terminal

The `ShiftWatch` GitHub Actions workflow provides a browser-based entry point:

1. Open the repository's **Actions** tab.
2. Select **ShiftWatch**.
3. Choose **Run workflow**.
4. Select `evaluate` or `monitor`; monitoring also accepts a target error rate.
5. After the run finishes, download the `shiftwatch-*` artifact from the run page.

Every push and pull request also runs the automated test suite. Manual runs execute tests first, then generate a CSV result and a JSON summary. Artifacts are retained for 14 days and require no credentials or external model API.

## Dataset format

One manually reviewed example per line:

```json
{"id":"example-001","text":"Scientists published a laboratory study.","label":"science"}
```

Allowed labels are `technology`, `business`, `science`, `campus`, and `other`. The demo data exists only to exercise the pipeline and must not be reported as a meaningful benchmark.

## Metrics

- **Accuracy/error rate:** task performance, counting abstention as unresolved.
- **Abstain rate:** how often the model declines to classify.
- **Brier score:** squared error between confidence and correctness; lower is better.
- **Error increase vs. clean:** degradation attributable to each controlled condition.
- **Detection alarms:** points at which CUSUM or EWMA flags sustained error growth.

## Responsible interpretation

The current dataset is tiny and synthetic, so results are software checks rather than scientific conclusions. A valid study will require a documented sampling procedure, independently reviewed labels, class-balance reporting, frozen test data, multiple model runs, uncertainty intervals, and qualitative review of failure cases.

## Roadmap

1. Build and document a 200–500 example human-reviewed evaluation set.
2. Add a local Ollama adapter with strict JSON parsing and retry accounting.
3. Add paraphrase and out-of-domain shifts without changing gold labels.
4. Report bootstrap confidence intervals and per-class false positives/negatives.
5. Simulate gradual and abrupt degradation; compare detection delay and false alarms.
6. Add GPU batch inference only when CPU inference becomes the bottleneck.
7. Add LoRA/QLoRA as an experiment and compare pre/post-tuning calibration, abstention, and shifted performance.

## Project positioning

ShiftWatch is best described as an AI reliability evaluation and monitoring project spanning reproducible data systems, trustworthy ML, and sequential change detection. Fine-tuning is a later experimental intervention, not evidence of alignment by itself.
