# context-budget-lab

Flagship. Benchmark suite answering: when should an agent use full context, retrieval, structured memory, summary memory, or prefix-cache-friendly prompting?

**Research question:** For agentic and RAG workloads, what context strategy gives the best quality/latency/cost frontier?

**Status:** in progress. The first runnable slice is live: tiny public-policy and synthetic-agent-memory datasets, five baseline context strategies, shared JSONL traces, summary output, and a real local Ollama smoke run.

Measures answer + citation accuracy alongside TTFT, TPOT/TBT, p50/p95, throughput, context tokens, cache hit rate, and cost per 1K useful answers. Datasets: public legal/policy docs, agent task logs, synthetic geo-audio transcripts (all public or synthetic, labeled).

## Quickstart

```bash
uv sync
uv run pytest
uv run python benchmarks/run_context_benchmark.py \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:3b \
  --limit 1 \
  --out results/local-smoke
```

Then summarize and plot the latest run:

```bash
uv run python analysis/summarize.py results/local-smoke/context-budget-20260612-004000
uv run python analysis/make_plots.py results/local-smoke/context-budget-20260612-004000
```

For CI or no-server checks:

```bash
uv run python benchmarks/run_context_benchmark.py --mock --limit 1 --out results/mock-smoke
```

## Baselines

- `full_context`: send every task-local source.
- `rag_topk`: lexical top-k retrieval over task-local sources.
- `summary_memory`: use compressed summary memory.
- `structured_memory`: use structured fact memory.
- `prefix_cache_friendly`: keep stable instructions before request-specific payload.

## First Smoke Artifact

Committed local run:

- `results/local-smoke/context-budget-20260612-004000/traces.jsonl`
- `results/local-smoke/context-budget-20260612-004000/run.json`
- `results/local-smoke/context-budget-20260612-004000/summary.json`
- `results/local-smoke/context-budget-20260612-004000/latency_quality.svg`

This run used Ollama's OpenAI-compatible endpoint on the M3 Pro with `qwen2.5:3b`, two toy tasks, five strategies, and zero request errors. It is a harness/reproducibility check, not a research claim.
