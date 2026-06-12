# PLAN — context-budget-lab (agent A)

You are **codex-A**, working only in this repo. Read `../AGENTS.md` (conventions, quality bar, ledger protocol) and append to `../agent-ledger.md` after every milestone. codex-B consumes `exports/*.jsonl` per `../workload-schema.md` — keep exports valid; any record-format change is a `CONTRACT CHANGE`.

## Status: v0 COMPLETE (2026-06-11) — current phase: v1

v0 shipped in commits `2a0754f`…`455bd57`: 40 tasks across two lanes, five-dimension deterministic scoring, 1,000-request local matrix on `qwen2.5:3b` (5 strategies × 40 tasks × 5 repeats, 0 errors), bootstrap-CI frontier (`analysis/frontier/`), `RESULTS.md`, and workload exports. Evidence: `RESULTS.md`, `uv run pytest` → 36 passed.

**v0 findings that drive v1** (details in `RESULTS.md`):

- Full context vs RAG top-k fact coverage is indistinguishable at N=5 on the 3b.
- `prefix_cache_friendly` is fastest but scored **0.100 on abstention** — the prompt shape appears to coerce an answer when memory is insufficient.
- Non-streaming requests mean TTFT ≈ total latency — **no TTFT/TBT claims are possible from v0**.
- Memory strategies cite plausibly while dropping required facts (answer/citation split keeps proving itself).

## v1 end goal

Upgrade v0 from "one small model, no TTFT" to a defensible cross-scale result: **does the strategy frontier shift between 3b and 7b, what does streaming TTFT actually show, and does the abstention failure have a fixable cause?**

## Milestones (each ends with: tests green, commit, ledger entry)

### A6 — Streaming client for real TTFT

- Switch the live client to streaming completions; record TTFT (first content chunk) and TBT/decode rate separately in traces per `../trace-schema.md`. Keep `--mock` non-streaming path working.
- Note: codex-B independently built a streaming Ollama adapter (`../inference-release-lab/infergate/live_runtime.py`). Do NOT import their code (repo boundary) — but the approach is proven; nonce rule still applies.
- Tests: mock streaming path produces distinct ttft/e2e fields; existing trace consumers (summarize/frontier) handle both old and new traces.

### A7 — Cross-scale matrix: qwen2.5:7b

- Rerun the full matrix on `qwen2.5:7b` (installed): 5 strategies × 40 tasks × 5 repeats, temp 0, seed recorded, nonce cache-busting. It will be slow on the M3 Pro — run it in chunks per strategy, commit summaries + one example trace dir per condition as you go.
- Extend `analysis/frontier.py` to plot both models on one frontier; add a per-strategy 3b-vs-7b delta table.
- The question to answer in `RESULTS.md`: which strategy gaps close (or open) with model scale, with CIs.

### A8 — Abstention failure experiment

- Hypothesis from v0: `prefix_cache_friendly`'s prompt shape coerces answers on missing-memory tasks (0.100 abstain-correct vs 0.833 for full context).
- Add ONE controlled prompt variant (explicit "say so if the memory does not contain this" instruction in the stable prefix) — same strategy otherwise. Run it on the memory lane only, both models, 5 repeats.
- Either outcome is a result: write it up in `RESULTS.md` (fixed → instruction placement matters and the cached-prefix benefit survives; not fixed → deeper issue, document it). Do not silently tune until it passes.

### A9 — Embedding retrieval variant (optional, flag first)

- `rag_topk_embed`: top-k via Ollama's embeddings endpoint (`nomic-embed-text`, free/local — pulling the model is allowed). Same k as lexical.
- This adds a strategy, not a dependency — use the existing HTTP client. If it needs a new Python package, `DECISION NEEDED:` first.
- Compare lexical vs embedding retrieval on both lanes; distractor-heavy tasks are where they should differ.

### A10 — RESULTS v1 + export refresh

- Rewrite `RESULTS.md` to cover both models, streaming TTFT, the abstention experiment, and (if run) embedding retrieval. Same claims discipline: every number cites its `results/...` path; CI-overlapping strategies stay "indistinguishable at this N".
- Re-run the exporter; confirm `tests/test_exports.py` still passes. If any task record changed, ledger-flag `CONTRACT CHANGE` review even if the schema didn't move.
- Update README status; append to the week's `../research-log/` entry.

## Boundaries (unchanged from v0)

No GPU/Modal/paid-API spend. No LLM judges. No new Python dependencies without a `DECISION NEEDED:` ledger line. No edits outside this repo except the ledger. Never fake live results — if Ollama is down, flag and do mock-path work.

## Review packet

Updated `RESULTS.md` + two-model frontier SVG + abstention experiment outcome + green pytest + ledger trail.
