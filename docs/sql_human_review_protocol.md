# SQL human-review protocol

This protocol turns qualitative inspection into a reproducible blinded evaluation. The reviewer should open only the generated packet until every selected row is complete. The separate key reveals the task id, perturbation condition, benchmark labels, and leakage terms and must remain closed during phase 1.

## What the reviewer fills in

- `human_concepts`: the concepts genuinely needed to explain the error, separated by `|`. Prefer labels from the benchmark vocabulary; add a proposed label in `notes` if the vocabulary is inadequate.
- `diagnosis_rating`: `fully_correct`, `partially_correct`, `incorrect`, or `ambiguous`.
- `leakage_rating`: `none`, `partial`, or `full`.
- `explanation_quality`: `0`, `1`, or `2`.
- `reviewer_confidence`: `1`, `2`, or `3`.
- `task_ambiguity`: `yes` or `no`.
- `notes`: a short reason for difficult decisions, missing concepts, false attributions, or dialect dependence.

Do not change the query, model output, or review id.

## Rubric

### Diagnosis

- `fully_correct`: identifies every material concept and makes no material false attribution.
- `partially_correct`: contains a useful correct diagnosis but misses a material concept, adds a material false concept, or gives an important misleading explanation.
- `incorrect`: does not identify a material concept or is predominantly misleading.
- `ambiguous`: the task cannot be judged reliably because its semantics depend on unstated schema, SQL dialect, or intent. Explain why in `notes`.

Synonyms should be judged by meaning, not exact wording. Record the normalized concept in `human_concepts`; do not penalize a correct explanation solely for using different prose.

### Answer leakage

- `none`: conceptual guidance only; it does not supply a directly usable repair.
- `partial`: reveals a crucial replacement operator, clause, or transformation, but not a complete corrected query.
- `full`: provides corrected SQL or instructions so specific that the answer can be copied with trivial editing.

Quoting the broken query is not leakage by itself. Judge what new repair information the response supplies.

### Explanation quality

- `0`: incorrect, harmful, or unusable.
- `1`: directionally useful but incomplete, vague, or partly misleading.
- `2`: clear, accurate, and appropriately conceptual without leaking the repair.

## Review procedure

1. Generate a packet with a fixed seed.
2. Move or hide the key and inspect only the packet.
3. Review one row at a time without searching for the source task.
4. Complete all required judgment fields. Use notes for every `ambiguous` rating and low-confidence decision.
5. Run the summarizer to unblind conditions and calculate human-adjudicated metrics.
6. Inspect disagreements between the human concepts and benchmark gold labels. Gold labels may be changed only after recording the rationale and versioning the dataset.

For a first study with one reviewer, blindly re-review a random 20% after at least seven days and report agreement. A stronger study uses two independent reviewers and a third-pass adjudication for disagreements. Never silently alter ratings after seeing the condition or gold key.

## Commands

```bash
python -m shiftwatch.cli sql-review-create \
  results/llama3_sql_smoke_v2.csv datasets/sql_misconceptions_30.jsonl \
  --packet reviews/generated/sql_smoke_packet.csv \
  --key reviews/generated/sql_smoke_key.csv --seed 7

# Fill the packet, then unblind and summarize.
python -m shiftwatch.cli sql-review-summarize \
  reviews/generated/sql_smoke_packet.csv reviews/generated/sql_smoke_key.csv \
  --output results/sql_human_review_summary.json
```

Generated packets and keys are ignored by Git because they can contain experiment outputs and because publicly exposing the key undermines blinding. Archive the completed packet, key, dataset commit, model/configuration, and summary together when freezing an experiment.
