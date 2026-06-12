# PLAN — context-budget-lab (agent A)

You are **codex-A**, working only in this repo. Read `../AGENTS.md` (conventions, quality bar, ledger protocol) and append to `../agent-ledger.md` after every milestone. codex-B consumes `exports/*.jsonl` per `../workload-schema.md` — existing export files must not change; v2 adds NEW files with new `dataset_id`s.

## Status: v0+v1 COMPLETE — current phase: v2 (confidence)

v0 (40 tasks, 3b matrix, frontier) and v1 (streaming TTFT, 7b cross-scale matrix, abstention experiment) are done; see `RESULTS.md` and ledger. v2 exists because the v1 outcomes are softer than they should be, for three diagnosed reasons:

1. **Statistics:** quality CIs are computed per-arm despite a paired design (same 40 tasks across strategies), and temp-0 repeats add no content variance — effective N is task count.
2. **No budget pressure:** contexts are 3–8 short snippets, so full-context is never stressed — the tradeoff the project is named after never engages.
3. **Uncalibrated instrument:** literal fact matching may under-credit paraphrase-heavy (larger) models; the 7b fact-coverage deficit is unresolved.

## Spend authorization (first paid API use this summer)

Human approved on 2026-06-12: up to **$10 total** for A12 + A14 via the Anthropic API (judge/subject model `claude-haiku-4-5-20251001`). Requires `ANTHROPIC_API_KEY` in the environment — if missing, ledger-flag `DECISION NEEDED:` and continue with $0 milestones. Record actual dollar spend in every ledger entry that uses the API and in the research log. Exceeding $10 requires a new approval.

## Pre-registered hypotheses (fixed BEFORE any v2 run; analysis reports each as supported / refuted / inconclusive)

- **H1:** RAG top-k's paired fact-coverage advantage over full-context increases with haystack size, and at 32 docs the paired delta excludes zero.
- **H2:** the 7b fact-coverage deficit vs 3b is a scoring artifact: judge-scored fact coverage shows 7b ≥ 3b on `full_context` and `rag_topk`. (If the judge confirms the deficit, H2 is refuted and the claim "scale didn't buy extraction here" is confirmed by two instruments.)
- **H3:** compression strategies (`summary_memory`, `structured_memory`) degrade fact coverage faster with haystack size than `rag_topk`.
- **Headline chart:** the haystack size at which full-context stops being Pareto-optimal on the quality/latency/cost frontier (the "crossing point").

## Scheduling constraint (shared local Ollama)

codex-B owns local Ollama for its deferred live phase-2 queue (B8/B9/B10). A11 and A12 are offline — do them now. A13/A14 live runs start only after codex-B posts live-queue completion in the ledger. If ambiguous, `DECISION NEEDED:`, don't race.

## Milestones (each ends with: tests green, commit, ledger entry)

### A11 — Paired-delta reanalysis ($0, existing traces only)

- Add `analysis/paired_deltas.py`: per-task deltas between strategy pairs (and 3b-vs-7b within strategy), bootstrap CIs over the 40 per-task deltas, for fact coverage / citation precision / citation recall / abstain-correct.
- Report which v1 "indistinguishable" verdicts become separations under pairing; update `RESULTS.md` with a paired-comparison table (clearly labeled as reanalysis of committed v1 traces, no new runs).
- Tests: gold cases on a tiny synthetic trace fixture — a pair that separates under pairing but not per-arm, and a null pair that stays null.

### A12 — Judge calibration of the fact scorer (~$2–5)

- `analysis/judge_rescore.py`: re-score fact coverage ONLY, on existing matrix + abstention traces. **Dedupe identical (task, answer-text) pairs first** — temp-0 repeats mostly collapse, cutting calls to roughly 400–800.
- Judge = `claude-haiku-4-5-20251001`, temperature 0, fixed versioned prompt (commit it): given question, gold answer, expected_facts, and model answer → per-fact JSON verdict {credited, reason}, paraphrase allowed. Citations are NOT judged (exact-ID matching is already the right instrument).
- Judge scores go in a separate `judge_scores.jsonl` keyed by request id — never overwrite deterministic fields.
- Deliverables: judge-vs-literal agreement table per strategy×model; H2 verdict with paired deltas under judge scoring; ~20 sampled disagreements in `analysis/judge_disagreements.md` for human spot-check; a proposed alias list (paraphrases the judge credited) in `analysis/proposed_fact_aliases.md` — **propose only, do not edit datasets**; flag `DECISION NEEDED:` for approval.
- Tests: judge-call mocked; parser + dedupe + agreement math tested without network.

### A13 — Context-size sweep ($0, the core v2 run; wait for Ollama release)

- Haystack builder: for each existing task, deterministic (seeded) haystack variants of {2, 8, 32} docs — gold docs always included, distractors sampled from other tasks' docs in the same lane. New dataset ids like `public_ai_policy_v2_h32`; original records untouched.
- Expand abstention tasks to ~20 (same rule: answer genuinely absent from memory), so abstention CIs stop being [0, 0.2]-wide.
- Run matrix: strategies {`full_context`, `rag_topk`, `summary_memory`, `structured_memory`} × sizes {2, 8, 32} × both models × repeats 3 (repeats are for latency; quality is paired over tasks). Skip the prefix-cache lane — its question (abstention/caching) is separate from the budget question.
- The 7b×32-doc cells will be slow on the M3 Pro — run one condition at a time, commit summaries + one example trace dir as you go, and ledger-log progress so wall-clock stalls are visible. If a full cell exceeds ~2h, halve repeats for that cell and say so.
- If alias list was approved by then, scorer uses it (aliases live next to datasets, versioned); otherwise run with v1 scorer and note it.
- Analysis: paired deltas everywhere; per-metric curves vs haystack size; the Pareto-crossing chart; H1/H3 verdicts with CIs.

### A14 — Optional API-subject anchor (~$2–4, within the same $10 cap)

- Run `claude-haiku-4-5-20251001` as a *subject* on the 32-doc config, 4 strategies, repeats 1. Question: do the local-model strategy rankings hold for a production-grade small model? One table in RESULTS; no sweeping claims from one config.

### A15 — RESULTS v2 + additive exports

- Rewrite `RESULTS.md` around the hypotheses: each H supported/refuted/inconclusive with its evidence path; updated limitations (what the judge did and did not fix); actual dollar spend.
- Export the h32 suites as NEW files (`exports/public_ai_policy_v2_h32.jsonl`, etc.) — additive long-context gate workloads for codex-B (their research question 3 names long context). Existing export files byte-identical; extend the export test to cover the new files. Ledger-flag `exports: v2 long-context suites available` (additive — not a CONTRACT CHANGE).
- README status update; research-log append with spend.

## Boundaries

- No GPU/Modal spend. Anthropic API only, within the $10 cap above; all other rules from `../AGENTS.md` hold (no new deps without flag, stay in repo, never fake results, keep negative results).
- Pre-registered hypotheses are frozen: if the data motivates a new hypothesis, it goes in RESULTS as exploratory, not as a pre-registered result.

## Review packet

Paired-comparison table + judge agreement/disagreement files + H1/H2/H3 verdicts + Pareto-crossing chart + spend ledger + green pytest. The question the packet answers: "which of the v1 maybe-findings are now real, and what did it cost to find out?"
