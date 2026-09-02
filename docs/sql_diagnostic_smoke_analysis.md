# SQL misconception diagnosis: smoke-test analysis

## Scope

This is an evaluation-pipeline smoke test, not a model benchmark. A local
`llama3:latest` model diagnosed the first five cases under four controlled
prompt conditions, for 20 diagnoses total. The 30-case starter set and its
concept labels have not yet received independent review.

## Measurement revision

The first run allowed arbitrary concept labels. The model produced reasonable
near-synonyms such as `nullability`, `aggregation`, and `subqueries`, while the
gold taxonomy used labels such as `null_semantics` and
`aggregation_semantics`. Exact matching therefore reported zero precision and
recall even when explanations discussed relevant ideas.

The revised prompt supplies the full global concept vocabulary without
revealing the labels for a particular task. This makes predictions comparable
while keeping the diagnosis problem nontrivial.

## Revised smoke results

| Condition | Micro precision | Micro recall | Leakage | Mean latency |
|---|---:|---:|---:|---:|
| clean | 0.50 | 0.33 | 0% | 3.40 s |
| paraphrase | 0.50 | 0.33 | 0% | 4.20 s |
| irrelevant context | 0.67 | 0.44 | 0% | 3.32 s |
| false premise | 0.50 | 0.33 | 0% | 3.25 s |

Four of five tasks received exactly the same predicted label set across all
conditions, for an attribution-consistency rate of 0.80. With only five tasks,
these values validate the implementation but do not support model comparisons.

## Leakage definition

A diagnosis is flagged when it provides nonempty corrected SQL, emits a code
block, or contains a task-specific repair fragment. This heuristic can produce
both false positives and false negatives and requires human adjudication before
research use.

## Drift connection

Each evaluation snapshot exports one recall error rate per concept. Ordered
snapshots for a fixed concept can be passed to `sql-monitor`, which applies the
existing CUSUM and EWMA detectors. The bundled synthetic history demonstrates a
stable 10% error period followed by degradation near 50%; it is not model data.

## Next study milestone

- independently review the 30 queries and gold labels;
- define label descriptions and adjudication rules;
- compare at least two model or prompt configurations;
- repeat stochastic configurations and report uncertainty;
- manually adjudicate concept disagreements and leakage flags;
- append ordered snapshots before interpreting CUSUM/EWMA alarms.
