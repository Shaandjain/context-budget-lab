# Context Budget Lab v1 Results

Reproduce the v1 analysis from committed traces:

```bash
uv run python analysis/frontier.py results/local-matrix --out-dir analysis/frontier --resamples 1000 --seed 1729
uv run python analysis/paired_deltas.py results/local-matrix --out-dir analysis/paired_deltas --resamples 1000 --seed 1729
uv run python analysis/abstention_variant.py --baseline-root results/local-matrix --variant-root results/abstention-variant --out-dir analysis/abstention_variant --dataset-id synthetic_agent_memory_v0 --models qwen2.5:3b,qwen2.5:7b
uv run python benchmarks/export_workloads.py
uv run pytest
```

## Research Question

Does the context-strategy frontier shift between `qwen2.5:3b` and `qwen2.5:7b`, what does streaming TTFT show once the client streams tokens, and is the prefix-cache abstention failure fixable with one explicit instruction?

## Setup

All runs used local Ollama on an M3 Pro, temperature 0, seed 1729, public or synthetic datasets only, and deterministic scoring. No GPU, Modal, paid API, embedding model, new Python dependency, or LLM judge was used.

The cross-scale matrix uses the existing 3b v0 traces and the new 7b streaming traces under `results/local-matrix`. Each model/strategy condition has 200 records and 0 request errors. The 3b matrix predates the streaming client, so `analysis/frontier/summary.md` intentionally reports `n/a` for 3b TTFT/TPOT. The 7b rows and A8 variant rows have real streaming `ttft_s`, `tpot_s`, and decode-rate fields.

## Cross-Scale Result

The 7b model did not broadly dominate the 3b model on fact coverage at this N. The per-strategy 7b-minus-3b deltas in `analysis/frontier/model_deltas.md` were: full context -0.060, RAG top-k -0.040, prefix-cache-friendly +0.003, structured memory +0.017, and summary memory -0.042.

The 7b model did improve several citation and schema measures on context-heavy strategies. From `analysis/frontier/model_deltas.md`, citation recall moved +0.129 for full context, +0.059 for RAG top-k, and +0.200 for prefix-cache-friendly; schema compliance moved +0.135 for full context, +0.070 for RAG top-k, and +0.165 for prefix-cache-friendly. Memory strategies did not get the same clean lift: summary memory schema moved -0.190 and structured memory schema moved -0.175.

The latency cost of scale is clear in the same delta table. 7b p50 end-to-end latency moved +3.083s for full context, +1.463s for RAG top-k, +3.112s for prefix-cache-friendly, +3.567s for structured memory, and +2.441s for summary memory.

## A11 Paired Reanalysis

The v1 frontier used per-arm bootstrap CIs, but the design is paired: the same tasks appear under every strategy and model. A11 reanalyzes the committed v1 traces by first averaging repeats per task, then bootstrapping over task-level deltas. No new model runs are included. Outputs live in `analysis/paired_deltas/summary.md`.

| comparison | metric | paired delta | A11 interpretation | source |
| --- | --- | --- | --- | --- |
| 3b `full_context` -> 3b `rag_topk` | fact coverage | -0.018 [-0.086, 0.040] | still indistinguishable | `analysis/paired_deltas/summary.md` |
| 7b `full_context` -> 7b `rag_topk` | fact coverage | +0.002 [-0.045, 0.053] | still indistinguishable | `analysis/paired_deltas/summary.md` |
| 3b `full_context` -> 3b `summary_memory` | fact coverage | -0.148 [-0.246, -0.053] | compression loss separates under pairing | `analysis/paired_deltas/summary.md` |
| 7b `rag_topk` -> 7b `summary_memory` | fact coverage | -0.132 [-0.227, -0.037] | compression loss separates under pairing | `analysis/paired_deltas/summary.md` |
| 7b `rag_topk` -> 7b `structured_memory` | fact coverage | -0.134 [-0.217, -0.047] | compression loss separates under pairing | `analysis/paired_deltas/summary.md` |
| 3b `full_context` -> 7b `full_context` | fact coverage | -0.060 [-0.131, 0.001] | 7b fact deficit still does not cleanly separate | `analysis/paired_deltas/summary.md` |
| 3b `rag_topk` -> 7b `rag_topk` | fact coverage | -0.040 [-0.100, 0.018] | 7b fact deficit still does not cleanly separate | `analysis/paired_deltas/summary.md` |
| 3b `full_context` -> 7b `full_context` | citation recall | +0.129 [0.029, 0.247] | 7b citation gain separates under pairing | `analysis/paired_deltas/summary.md` |
| 3b `rag_topk` -> 7b `rag_topk` | citation precision | +0.095 [0.013, 0.177] | 7b citation gain separates under pairing | `analysis/paired_deltas/summary.md` |

This sharpens the v1 takeaway. RAG top-k and full-context remain tied on measured fact coverage for both model scales. The memory-compression losses are now stronger evidence, because several comparisons that had overlapping per-arm CIs become paired separations. The 7b fact-coverage deficit remains close to zero rather than decisive, while 7b citation gains survive the paired design.

## Streaming Timing

The 7b matrix now separates first-token and decode timing. From `analysis/frontier/summary.md`, RAG top-k had p50 TTFT 1.059s, mean TPOT 0.030s, and p50 end-to-end latency 4.204s. Full context had p50 TTFT 1.847s, mean TPOT 0.035s, and p50 latency 5.148s. Prefix-cache-friendly had p50 TTFT 1.331s, mean TPOT 0.038s, and p50 latency 5.032s.

This fixes the v0 measurement issue for new live runs: TTFT is no longer just total request latency. It does not retroactively make the 3b matrix a TTFT result; those rows remain `n/a` for streaming timing in `analysis/frontier/summary.md`.

## Abstention Variant

A8 added one controlled variant, `prefix_cache_abstain`: same lexical top-k retrieval and same request-specific payload as `prefix_cache_friendly`, but the stable prefix explicitly says to report insufficient evidence instead of guessing. The comparison in `analysis/abstention_variant/summary.md` filters both baseline and variant to `synthetic_agent_memory_v0`.

| model | baseline abstain | variant abstain | delta | baseline useful | variant useful | source |
| --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:3b | 0.100 | 0.000 | -0.100 | 38 | 40 | `analysis/abstention_variant/summary.md` |
| qwen2.5:7b | 0.200 | 0.633 | +0.433 | 48 | 25 | `analysis/abstention_variant/summary.md` |

The variant is not a clean fix. On 3b it made abstention worse. On 7b it improved abstention correctness, but fact coverage fell 0.657 -> 0.557, schema compliance fell 0.950 -> 0.810, and useful answers dropped 48 -> 25, all from `analysis/abstention_variant/summary.md`. Instruction placement matters, but the prefix-cache failure is not solved by adding one sentence.

## Negative Results / Surprises

The larger model mostly improved citation behavior, not answer coverage. That matters for agent memory systems: better-looking citations can still hide missing facts.

Prefix-cache-friendly remained the fastest-looking and most cacheable prompt shape in intent, but the original abstention failure persisted on 7b: `analysis/frontier/summary.md` reports abstain-correct 0.200 for 7b prefix-cache-friendly, versus 0.733 for 7b full context.

The optional embedding retrieval variant was not run. `nomic-embed-text` was not installed locally, and adding embedding retrieval would require new embedding endpoint plumbing plus another experiment matrix. A10 therefore reports lexical retrieval only.

## Limitations

This is a local quantized Ollama result, not a serving-infrastructure benchmark. End-to-end latency is useful for this machine, but it should not be generalized to GPU serving.

Bootstrap confidence intervals in `analysis/frontier/summary.md` are over repeated task traces, not a guarantee about broader workloads. The task set is 40 controlled tasks split across public-policy and synthetic-memory lanes.

The A11 paired reanalysis in `analysis/paired_deltas/summary.md` is stronger for within-task comparisons, but it does not fix the literal scorer limitation or add long-context budget pressure. Those are A12 and A13's jobs.

The scorer is deterministic and literal. It catches regressions consistently, but it will miss valid paraphrases and cannot judge reasoning quality beyond the expected facts/citations/schema checks.

The workload exports remain producer-side task suites for `../inference-release-lab`; no record-format or schema change was made in v1.

## Review Packet

- Cross-scale frontier: `analysis/frontier/frontier.svg`
- Cross-scale summary: `analysis/frontier/summary.md`
- 7b-minus-3b deltas: `analysis/frontier/model_deltas.md`
- Paired task-level deltas: `analysis/paired_deltas/summary.md`
- Abstention experiment: `analysis/abstention_variant/summary.md`
- Raw traces: `results/local-matrix/**/traces.jsonl` and `results/abstention-variant/**/traces.jsonl`
- Workload exports: `exports/public_ai_policy_v0.jsonl`, `exports/synthetic_agent_memory_v0.jsonl`
