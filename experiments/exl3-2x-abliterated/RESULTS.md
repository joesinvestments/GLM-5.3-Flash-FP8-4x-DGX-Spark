# AE Hermes A/B — 2026-08-30

## Method

The comparison used the same `glm53ae20` Hermes profile, native high
reasoning, temperature 0, 16 AE read tools, and the same frozen repaired
20-job suite. Telegram and Buzz were disabled. The benchmark allowed no CRM
writes. Raw CRM traces remain in the private local artifact directory and are
not committed here.

The inherited runner writes a stale `medium-default-no-cli-override` label
into result JSON. That label is metadata only: the frozen profile snapshot and
live config both set `agent.reasoning_effort: high` for both runs.

## Results

| Measure | Four-Spark FP8/SGLang adaptive NEXTN | Two-Spark EXL3/vLLM DFlash2, Abliterated-on |
|---|---:|---:|
| Completed AE jobs | 19/20 | 20/20 |
| Harness timeouts | 1 | 0 |
| Compliant / partial / failed jobs | 19 / 0 / 1 | 17 / 2 / 1 |
| Malformed tool calls | 0 | 0 |
| Tool-result errors | 0 | 1 |
| CRM writes | 0 | 0 |
| Tool calls | 77 | 81 |
| Length-finish continuations | 0 | 13 |
| Total suite time | 930.018 s | 1,543.373 s (+66.0%) |
| Median job time | 34.341 s | 68.764 s (+100.2%) |
| Structured decode median | 73.255 tok/s | 66.566 tok/s (-9.1%) |
| Structured median TTFT | 0.403 s | 0.460 s (+14.3%) |
| Prose decode median | 39.329 tok/s | 26.986 tok/s (-31.4%) |
| Prose median TTFT | 0.355 s | 0.453 s (+27.5%) |

The control timeout was job 20, before its first tool call or final answer.
Jobs 1-19 completed with the required AE identity gate, valid tool JSON, no
tool errors, and no writes. This fresh control is the comparison point; older
qualified-run counts are not substituted for it.

The candidate used half as many Sparks, so its per-Spark decode efficiency was
about 81.7% higher on the structured ruler and 37.2% higher on prose. That is a
capacity-efficiency benefit, not an interactive-speed win: one request was
slower on both matched rulers. DFlash2 acceptance explains the workload split.
The structured median accepted 95.6% of draft tokens (6.692 per step), while
prose accepted 34.9% (2.440 per step).

## Decision

Do not promote this candidate. It saves two Sparks and completed the long
self-score job that timed out on the control, but it doubled median AE latency
and failed one exact tool-use contract.

Manual grading of the candidate found 17 passes, two partials, and one fail:

- Job 14 failed its exact-once reducer contract. vLLM emitted a
  `chatcmpl-tool-*` call ID, the fixed reducer rejected that spillover filename,
  and the model improvised two more `execute_code` calls, including a hard-link
  workaround. No sensitive value leaked and no CRM write occurred, but the
  prescribed safe path was not followed.
- Job 15 was partial because it emitted note-field presence categories outside
  the prompt's allowed phone/address categorical reducer shape. It emitted no
  raw contact value.
- Job 20 was partial because it claimed job 12's table was truncated and
  contaminated by another job. The saved job 12 output is complete and clean.

The tool-ID mismatch belongs in the vLLM recipe layer, not Hermes: a future
repair should make vLLM's GLM parser emit the `call_*` identifier form already
used by the qualified SGLang lane, then rerun job 14 before any full retest.
That repair would not change this A/B verdict because latency and licensing
still block promotion.
