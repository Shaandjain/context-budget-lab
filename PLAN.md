# PLAN — context-budget-lab (agent A)

You are **codex-A**, working only in this repo. Read `../AGENTS.md` (conventions, quality bar, ledger protocol) and append to `../agent-ledger.md` after every milestone. codex-B consumes `exports/*.jsonl` per `../workload-schema.md` — existing export files must not change; v2 adds NEW files with new `dataset_id`s.

## Status: v0+v1 COMPLETE — v2 A15 review packet COMPLETE

v0 (40 tasks, 3b matrix, frontier) and v1 (streaming TTFT, 7b cross-scale matrix, abstention experiment) are done; see `RESULTS.md` and ledger. v2 exists because the v1 outcomes are softer than they should be, for three diagnosed reasons:

1. **Statistics:** quality CIs are computed per-arm despite a paired design (same 40 tasks across strategies), and temp-0 repeats add no content variance — effective N is task count.
2. **No budget pressure:** contexts are 3–8 short snippets, so full-context is never stressed — the tradeoff the project is named after never engages.
3. **Uncalibrated instrument:** literal fact matching may under-credit paraphrase-heavy (larger) models; the 7b fact-coverage deficit is unresolved.

A11 paired reanalysis and A12 judge calibration are complete. A13a/A13b/A13c are complete: the v2 h{2,8,32} haystack datasets exist, the abstention suite is expanded to 20 tasks, and the mock matrix proved the four-strategy path before GPU spend. A13d started on Modal, then stopped when the Modal workspace billing-cycle cap reclaimed the endpoint and returned 429s. The interrupted traces are diagnostic only and are archived under `results/v2-matrix-interrupted-modal-cap-20260624/`.

After the cap was cleared, A13d resumed on Modal L4 and completed a clean final matrix under `results/v2-matrix/`: 2 models x 4 strategies x all nine v2 h{2,8,32} datasets x 3 repeats. Every final condition has 540 traces and zero request errors. A13e analysis is committed under `analysis/v2_sweep/` and reports the matrix clean before hypothesis summaries. Interrupted/errored diagnostics remain outside final evidence under `results/v2-matrix-interrupted-modal-cap-20260624/`, `results/v2-matrix-paused-20260703/`, and `results/v2-matrix-interrupted-modal-transient-20260706/`. A15 closes Project 2 with the review packet in `RESULTS.md`, additive h32 exports for release-lab, and green local verification.

## Spend authorization (first paid API use this summer)

Human approved on 2026-06-12: up to **$10 total** for A12 + A14 via the Anthropic API (judge/subject model `claude-haiku-4-5-20251001`). Requires `ANTHROPIC_API_KEY` in the environment — if missing, ledger-flag `DECISION NEEDED:` and continue with $0 milestones. Record actual dollar spend in every ledger entry that uses the API and in the research log. Exceeding $10 requires a new approval.

## Modal authorization and restart gate

Human approved Modal GPU spend for A13 on 2026-06-24, separately from the Anthropic cap, and reconfirmed on 2026-07-02 that A13 should remain authoritative on Modal GPU rather than falling back to local Ollama. Before any new paid run:

1. Confirm no `cbl-vllm-*` Modal apps are running.
2. Confirm the Modal workspace cap is cleared by deploying one endpoint and checking `/v1/models`.
3. Confirm the served model name matches the intended cell (`qwen2.5-3b-instruct` or `qwen2.5-7b-instruct`).
4. Run one smoke request, then stop if the endpoint returns billing/cap/429 errors.

If any billing/cap/429 error appears, stop the run, stop all `cbl-vllm-*` apps, append a ledger `DECISION NEEDED:`, and do not mix those traces into final A13 analysis.

## Pre-registered hypotheses (fixed BEFORE any v2 run; analysis reports each as supported / refuted / inconclusive)

- **H1:** RAG top-k's paired fact-coverage advantage over full-context increases with haystack size, and at 32 docs the paired delta excludes zero.
- **H2:** the 7b fact-coverage deficit vs 3b is a scoring artifact: judge-scored fact coverage shows 7b ≥ 3b on `full_context` and `rag_topk`. (If the judge confirms the deficit, H2 is refuted and the claim "scale didn't buy extraction here" is confirmed by two instruments.)
- **H3:** compression strategies (`summary_memory`, `structured_memory`) degrade fact coverage faster with haystack size than `rag_topk`.
- **Headline chart:** the haystack size at which full-context stops being Pareto-optimal on the quality/latency/cost frontier (the "crossing point").

## Scheduling constraint

codex-B released local Ollama on 2026-06-12. A13 should not use the local Ollama fallback unless the human explicitly changes the compute decision again; the current plan is Modal-only for final v2 evidence.

## Milestones (each ends with: tests green, commit, ledger entry)

### A11 — Paired-delta reanalysis ($0, existing traces only) — DONE

- Add `analysis/paired_deltas.py`: per-task deltas between strategy pairs (and 3b-vs-7b within strategy), bootstrap CIs over the 40 per-task deltas, for fact coverage / citation precision / citation recall / abstain-correct.
- Report which v1 "indistinguishable" verdicts become separations under pairing; update `RESULTS.md` with a paired-comparison table (clearly labeled as reanalysis of committed v1 traces, no new runs).
- Tests: gold cases on a tiny synthetic trace fixture — a pair that separates under pairing but not per-arm, and a null pair that stays null.

### A12 — Judge calibration of the fact scorer (~$2–5) — DONE

- `analysis/judge_rescore.py`: re-score fact coverage ONLY, on existing matrix + abstention traces. **Dedupe identical (task, answer-text) pairs first** — temp-0 repeats mostly collapse, cutting calls to roughly 400–800.
- Judge = `claude-haiku-4-5-20251001`, temperature 0, fixed versioned prompt (commit it): given question, gold answer, expected_facts, and model answer → per-fact JSON verdict {credited, reason}, paraphrase allowed. Citations are NOT judged (exact-ID matching is already the right instrument).
- Judge scores go in a separate `judge_scores.jsonl` keyed by request id — never overwrite deterministic fields.
- Deliverables: judge-vs-literal agreement table per strategy×model; H2 verdict with paired deltas under judge scoring; ~20 sampled disagreements in `analysis/judge_disagreements.md` for human spot-check; a proposed alias list (paraphrases the judge credited) in `analysis/proposed_fact_aliases.md` — **propose only, do not edit datasets**; flag `DECISION NEEDED:` for approval.
- Tests: judge-call mocked; parser + dedupe + agreement math tested without network.

### A13 — Context-size sweep (Modal GPU, core v2 run) — DONE

- DONE: Haystack builder generated deterministic variants of {2, 8, 32} docs, with gold docs always included. Because the v0 lanes did not have enough unique same-lane distractors for true 32-doc haystacks, h32 draws non-answering distractors from the union of both lanes; report this cross-lane padding caveat in RESULTS.
- DONE: Abstention expanded to 20 tasks in `synthetic_agent_memory_abstain_v2` and swept across h{2,8,32}.
- DONE: A mock matrix over the v2 datasets proved the four-strategy path before GPU spend.
- DONE: interrupted Modal cap traces are diagnostic only. Final evidence is a fresh clean matrix: strategies {`full_context`, `rag_topk`, `summary_memory`, `structured_memory`} × sizes {2, 8, 32} × datasets {policy, memory, abstain} × models {`qwen2.5-3b-instruct`, `qwen2.5-7b-instruct`} × repeats 3. All eight final model/strategy cells have 540 traces and zero request errors.
- DONE: ran one model/strategy condition at a time, wrote summaries immediately, verified expected record count (540) and zero request errors before moving on, and stopped all Modal apps after each model block. One 7B `structured_memory` attempt hit transient connection reset/timeout errors and is archived under `results/v2-matrix-interrupted-modal-transient-20260706/`; the clean rerun is the only `structured_memory` run under final `results/v2-matrix/`.
- **Alias list REJECTED (2026-06-14, see decisions.md): run A13 on the v1 deterministic scorer.** Do not wire `proposed_fact_aliases.md` into `scoring.py`. Report judge fact-coverage as a separate column beside the literal column where relevant; the literal/judge gap is the calibration finding, not a scorer patch. If the morphology fix below lands first, use that scorer version and label it.
- Optional separate milestone (not the alias table): fix the literal matcher's morphology/word-order brittleness (`benchmark` not matching `benchmarks`, `responsible use guidance` vs `guidance on responsible use`) as a *general* normalization rule applied to all tasks, with gold **and negative** tests, then re-run and report literal-v2 vs literal-v1 vs judge. Reproducible; the curated per-answer alias list is not.
- DONE: `uv run python analysis/v2_sweep.py results/v2-matrix --out-dir analysis/v2_sweep --resamples 1000 --seed 1729` reports `Clean matrix: True`, completeness first, paired H1/H3 deltas by haystack size, and the Pareto frontier table. V2 claims should cite `analysis/v2_sweep/summary.md` and not the archived interrupted/errored cells.

### A14 — Optional API-subject anchor (~$2–4, within the same $10 cap) — SKIPPED FOR CLOSEOUT

- Skipped for A15 closeout. Do not run the optional Anthropic subject-model anchor unless the human explicitly reopens it later. Project 2 closes on the committed local/Modal evidence packet.

### A15 — RESULTS v2 + additive exports — DONE

- DONE: h32 suites are exported as NEW files (`exports/public_ai_policy_v2_h32.jsonl`, etc.) and existing v0 export files stayed byte-identical.
- DONE: README status update and research-log append.
- DONE: `RESULTS.md` has a v2 A13/A15 section with H1/H2/H3 verdicts, frontier summary, limitations, review packet pointers, and spend caveat. Historical v1/A11/A12 sections remain below it as context, not the current headline.

## Boundaries

- Modal GPU spend is allowed only for A13 under the approval and preflight gate above. Anthropic API is allowed only within the $10 A12/A14 cap. No new dependencies without flag, stay in repo except explicitly requested workspace ledger/log/decision updates, never fake results, keep negative results.
- Pre-registered hypotheses are frozen: if the data motivates a new hypothesis, it goes in RESULTS as exploratory, not as a pre-registered result.

## Review packet

A15 packet: clean-matrix completeness table + H1/H2/H3 verdicts + frontier table + additive h32 exports + A11 paired table + A12 judge calibration/disagreement files + spend ledger + green pytest. The question the packet answers: "which of the v1 maybe-findings are now real, and what did it cost to find out?"
