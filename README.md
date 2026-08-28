# ShiftWatch

ShiftWatch is a reproducible evaluation pipeline for studying how model reliability changes under controlled input distribution shifts. It measures failures on clean and perturbed text, then provides sequential detectors for monitoring degradation over time.

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
python -m shiftwatch.cli datasets/demo.jsonl --output results/evaluation.csv
python -m unittest discover -v
```

The CLI prints a JSON summary and writes one row per example and condition. Fixed random seeds make perturbations reproducible.

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
