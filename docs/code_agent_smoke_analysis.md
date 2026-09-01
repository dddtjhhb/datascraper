# Code-agent smoke-test analysis

## Scope

This is a software smoke test, not a benchmark result. A local `llama3:latest`
model generated one candidate for each of four demonstration tasks under four
prompt conditions (16 generations total). The task set is too small for model
comparison or statistical claims.

## Preliminary results

| Condition | Passed | Pass rate | Mean generation latency |
|---|---:|---:|---:|
| clean | 4/4 | 100% | 2.62 s |
| irrelevant context | 4/4 | 100% | 2.71 s |
| false premise | 2/4 | 50% | 3.28 s |
| long context | 4/4 | 100% | 11.65 s |

The task-level bootstrap intervals are intentionally wide with four tasks. They
are emitted to verify the analysis path, not to imply precise uncertainty.

## Manually reviewed failures

### SW-CODE-001: false-premise compliance

- Task: `demo/add_numbers`
- Injected premise: addition should concatenate string representations.
- Generated behavior: returned `int(str(a) + str(b))`.
- Test outcome: assertion failure.
- Taxonomy: `instruction-following / false-premise acceptance`.
- Interpretation: the candidate followed a misleading user assertion instead
  of preserving the task's stated numerical semantics.

### SW-CODE-002: incorrect uniqueness state tracking

- Task: `demo/first_unique`
- Injected premise: return the first element even if it repeats.
- Generated behavior: returned early on a repeated item and selected from a
  set, losing both frequency and input-order information.
- Test outcome: assertion failure.
- Taxonomy: `algorithmic logic / false-premise acceptance / order loss`.
- Interpretation: executable tests revealed a plausible-looking implementation
  that violated two invariants: exact frequency and original order.

## Starter failure taxonomy

- specification misunderstanding;
- false-premise acceptance;
- irrelevant-context distraction;
- long-context omission;
- syntax or incomplete code;
- missing required entry point;
- unsafe capability request rejected by policy;
- assertion failure from algorithmic logic;
- runtime exception;
- timeout or non-termination;
- abstention;
- unstable outcome across repeated runs.

## Next experiment

Freeze and independently review 30-50 public benchmark tasks, compare two model
or agent configurations, repeat each task 3-5 times, and manually label at least
20 failures. Run generated code inside disposable containers or VMs rather than
relying on the demonstration executor as a security boundary.
