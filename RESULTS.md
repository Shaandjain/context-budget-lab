# Context Budget Lab Results

Reproduce the committed local analyses:

```bash
uv run python analysis/frontier.py results/local-matrix --out-dir analysis/frontier --resamples 1000 --seed 1729
uv run python analysis/paired_deltas.py results/local-matrix --out-dir analysis/paired_deltas --resamples 1000 --seed 1729
uv run python analysis/abstention_variant.py --baseline-root results/local-matrix --variant-root results/abstention-variant --out-dir analysis/abstention_variant --dataset-id synthetic_agent_memory_v0 --models qwen2.5:3b,qwen2.5:7b
uv run python benchmarks/export_workloads.py
uv run pytest
```

The A12 judge-calibration artifacts are committed under `analysis/judge_rescore/`. Re-running the judge from scratch requires `ANTHROPIC_API_KEY` and spends Anthropic API budget; the committed run used `claude-haiku-4-5-20251001` and spent $2.071164 total.

Reproduce the A13 v2 Modal analysis:

```bash
uv run python analysis/v2_sweep.py results/v2-matrix --out-dir analysis/v2_sweep --resamples 1000 --seed 1729
```

## Latest: A13 v2 Modal Sweep

A13 is the first version of this lab that actually stresses the context budget. It runs both `qwen2.5-3b-instruct` and `qwen2.5-7b-instruct` on Modal L4 over nine v2 datasets: policy, memory, and abstention tasks at h2, h8, and h32 haystack sizes. Each final model/strategy condition has 540 traces and zero request errors. The analyzer reports `Clean matrix: True` in `analysis/v2_sweep/summary.md`.

One 7B `structured_memory` attempt hit transient connection reset/read-timeout errors on a single h32 memory case and is archived under `results/v2-matrix-interrupted-modal-transient-20260706/`. It is not part of final analysis; the clean rerun is `results/v2-matrix/qwen2-5-7b-instruct/structured_memory/context-budget-20260706-223401`.

| model | strategy | traces | errors | fact coverage | abstain correct | avg latency s | avg input tokens | useful answers | source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen2.5-3b-instruct` | `full_context` | 540 | 0 | 0.554 | 0.865 | 4.9711 | 999.03 | 169 | `results/v2-matrix/qwen2-5-3b-instruct/full_context/context-budget-20260703-022226/summary.json` |
| `qwen2.5-3b-instruct` | `rag_topk` | 540 | 0 | 0.596 | 0.397 | 4.9070 | 288.08 | 216 | `results/v2-matrix/qwen2-5-3b-instruct/rag_topk/context-budget-20260703-030945/summary.json` |
| `qwen2.5-3b-instruct` | `summary_memory` | 540 | 0 | 0.589 | 0.862 | 3.0786 | 671.99 | 92 | `results/v2-matrix/qwen2-5-3b-instruct/summary_memory/context-budget-20260706-172537/summary.json` |
| `qwen2.5-3b-instruct` | `structured_memory` | 540 | 0 | 0.479 | 0.332 | 3.3169 | 750.26 | 89 | `results/v2-matrix/qwen2-5-3b-instruct/structured_memory/context-budget-20260706-175353/summary.json` |
| `qwen2.5-7b-instruct` | `full_context` | 540 | 0 | 0.579 | 0.895 | 5.0369 | 999.03 | 221 | `results/v2-matrix/qwen2-5-7b-instruct/full_context/context-budget-20260706-182714/summary.json` |
| `qwen2.5-7b-instruct` | `rag_topk` | 540 | 0 | 0.576 | 0.850 | 5.5955 | 288.08 | 197 | `results/v2-matrix/qwen2-5-7b-instruct/rag_topk/context-budget-20260706-191451/summary.json` |
| `qwen2.5-7b-instruct` | `summary_memory` | 540 | 0 | 0.611 | 0.900 | 5.7753 | 671.99 | 95 | `results/v2-matrix/qwen2-5-7b-instruct/summary_memory/context-budget-20260706-200602/summary.json` |
| `qwen2.5-7b-instruct` | `structured_memory` | 540 | 0 | 0.495 | 0.793 | 5.5537 | 750.26 | 104 | `results/v2-matrix/qwen2-5-7b-instruct/structured_memory/context-budget-20260706-223401/summary.json` |

## A13 Hypothesis Verdicts

**H1: RAG's fact-coverage advantage over full context increases with haystack size.** Inconclusive for both models. For 3B, RAG is directionally above full context at h2/h8/h32, but h32 is +0.029 with CI crossing zero. For 7B, the h32 delta is +0.011 with CI crossing zero. Source: `analysis/v2_sweep/summary.md`.

**H2: the 7B fact-coverage deficit is a scoring artifact.** Still inconclusive from A12 judge calibration, not rescued by A13. The judge raised fact coverage across rows, confirming literal under-crediting, but judge-scored 7B-minus-3B paired deltas stayed slightly negative with CIs crossing zero for `full_context` and `rag_topk`. Source: `analysis/judge_rescore/summary.md`.

**H3: compression strategies degrade faster with haystack size than RAG.** Supported for 7B `structured_memory`; inconclusive for 3B `structured_memory` and both `summary_memory` rows. The practical read is sharper than the formal trend test: `structured_memory` is consistently below RAG on fact coverage for both models, while `summary_memory` stays competitive and is the strongest compression candidate. Source: `analysis/v2_sweep/summary.md`.

## A13 Frontier Read

At h32, RAG gives the strongest token reduction: mean input tokens are about 312 versus 2,145 for full context on both models. But quality does not collapse for full context on these 32-document synthetic/public workloads, so there is no clean "full context crosses off the frontier" story yet.

The clearest product-shaped result is that strategy choice is model-sensitive:

- On 3B, `rag_topk` and `summary_memory` are the useful frontier strategies; full context is not Pareto-optimal at h2/h8/h32 in `analysis/v2_sweep/summary.md`.
- On 7B, the frontier is flatter because latency and quality are close across several strategies. `summary_memory` has the best aggregate fact coverage and abstention correctness, but fewer useful answers than full context or RAG.
- `structured_memory` is not earning its complexity yet. It compresses, but fact coverage lags: 3B `structured_memory - rag_topk` is about -0.12 at every haystack size, and 7B is -0.082/-0.071/-0.090 for h2/h8/h32.

A13 Modal spend has not been read from the dashboard in this repo. The ledger records the operational spend checkpoints, and all Modal apps were stopped after the final run.

## Research Question

Does the context-strategy frontier shift between `qwen2.5:3b` and `qwen2.5:7b`, what does streaming TTFT show once the client streams tokens, and is the prefix-cache abstention failure fixable with one explicit instruction?

## Setup

The v1 model runs used local Ollama on an M3 Pro, temperature 0, seed 1729, public or synthetic datasets only, and deterministic scoring. No GPU, Modal, embedding model, or new Python dependency was used. A12 adds one paid Anthropic API judge calibration pass over saved traces only; it does not add new subject-model answers and does not judge citations.

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

## A12 Judge Calibration

A12 re-scored fact coverage only with `claude-haiku-4-5-20251001`, temperature 0, prompt version `fact-coverage-judge-v1`. It deduped identical `(task, answer excerpt)` pairs before judging: 2,200 request rows collapsed to 866 unique judge calls. Outputs live in `analysis/judge_rescore/summary.md`, `analysis/judge_rescore/judge_scores.jsonl`, `analysis/judge_rescore/judge_disagreements.md`, and `analysis/judge_rescore/proposed_fact_aliases.md`.

The judge broadly confirms that the literal scorer under-credits paraphrase. Judge fact coverage is higher than literal fact coverage for every strategy/model row, with deltas from +0.042 to +0.210. Per-fact agreement ranges from 0.764 to 0.883, and the sampled disagreements show exactly the expected failure mode: phrases like "people should know when automated systems are being used" credited for "interacting with an AI system," or "powerful dual-use models" credited for "dual-use foundation models."

H2 remains **inconclusive**, not supported. Judge-scored 7b-minus-3b paired deltas are still slightly negative on the two pre-registered comparisons, but both CIs cross zero:

| comparison | judge paired delta | H2 interpretation | source |
| --- | --- | --- | --- |
| 3b `full_context` -> 7b `full_context` | -0.021 [-0.105, +0.058] | inconclusive | `analysis/judge_rescore/summary.md` |
| 3b `rag_topk` -> 7b `rag_topk` | -0.024 [-0.096, +0.047] | inconclusive | `analysis/judge_rescore/summary.md` |

So the judge did fix part of the instrument problem, but it did not flip the scale result into "7b >= 3b" on fact coverage. The safest claim is: scaling improved citation discipline, and judge scoring reduces literal-match undercounting, but this workload still does not show a fact-coverage win from 7b. A12 API spend was $2.071164 total, including $0.647025 from an unrecovered first pass after fixing a non-global `request_id` collision in the score writer; the final score file has all 2,200 rows.

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

The A11 paired reanalysis in `analysis/paired_deltas/summary.md` is stronger for within-task comparisons, but it does not add long-context budget pressure. That is A13's job.

The deterministic scorer is literal. A12 shows that it misses valid paraphrases, but the judge is still only a calibration instrument over saved `answer_excerpt` fields, not full model answers and not ground truth. The judge does not evaluate citations, formatting, or reasoning quality beyond expected fact coverage.

The workload exports remain producer-side task suites for `../inference-release-lab`; no record-format or schema change was made in v1.

## Review Packet

- Cross-scale frontier: `analysis/frontier/frontier.svg`
- Cross-scale summary: `analysis/frontier/summary.md`
- 7b-minus-3b deltas: `analysis/frontier/model_deltas.md`
- Paired task-level deltas: `analysis/paired_deltas/summary.md`
- Judge fact-coverage calibration: `analysis/judge_rescore/summary.md`
- Judge disagreement sample: `analysis/judge_rescore/judge_disagreements.md`
- Proposed fact aliases for review: `analysis/judge_rescore/proposed_fact_aliases.md`
- Abstention experiment: `analysis/abstention_variant/summary.md`
- Raw traces: `results/local-matrix/**/traces.jsonl` and `results/abstention-variant/**/traces.jsonl`
- Workload exports: `exports/public_ai_policy_v0.jsonl`, `exports/synthetic_agent_memory_v0.jsonl`
