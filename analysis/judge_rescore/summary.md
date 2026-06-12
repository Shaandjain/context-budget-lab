# Judge Fact-Coverage Calibration

Judge model: `claude-haiku-4-5-20251001`. Prompt version: `fact-coverage-judge-v1`.
Answer text source: `meta.answer_excerpt`. These traces store saved answer excerpts, not full unrecoverable answers.
Requests scored: 2200; unique judge calls: 866; total API spend: $2.071164.
Spend includes $0.647025 from an unrecovered first pass; score-file calls account for $1.424139.

## Agreement

| model | strategy | requests | unique cases | literal fact coverage | judge fact coverage | judge - literal | per-fact agreement | judge credited/literal missed | literal credited/judge missed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:3b | full_context | 200 | 64 | 0.785 | 0.843 | +0.058 | 0.844 | 80 | 45 |
| qwen2.5:3b | prefix_cache_abstain | 100 | 29 | 0.527 | 0.654 | +0.127 | 0.847 | 54 | 5 |
| qwen2.5:3b | prefix_cache_friendly | 200 | 75 | 0.730 | 0.859 | +0.129 | 0.865 | 102 | 6 |
| qwen2.5:3b | rag_topk | 200 | 99 | 0.767 | 0.849 | +0.082 | 0.815 | 102 | 46 |
| qwen2.5:3b | structured_memory | 200 | 81 | 0.576 | 0.632 | +0.055 | 0.764 | 111 | 78 |
| qwen2.5:3b | summary_memory | 200 | 74 | 0.636 | 0.679 | +0.042 | 0.869 | 65 | 40 |
| qwen2.5:7b | full_context | 200 | 73 | 0.725 | 0.822 | +0.098 | 0.848 | 94 | 28 |
| qwen2.5:7b | prefix_cache_abstain | 100 | 36 | 0.557 | 0.767 | +0.210 | 0.800 | 77 | 0 |
| qwen2.5:7b | prefix_cache_friendly | 200 | 66 | 0.733 | 0.819 | +0.086 | 0.846 | 90 | 33 |
| qwen2.5:7b | rag_topk | 200 | 80 | 0.727 | 0.825 | +0.098 | 0.831 | 100 | 35 |
| qwen2.5:7b | structured_memory | 200 | 100 | 0.593 | 0.697 | +0.104 | 0.882 | 83 | 11 |
| qwen2.5:7b | summary_memory | 200 | 95 | 0.595 | 0.658 | +0.063 | 0.866 | 75 | 32 |

## H2 Verdict

Verdict: **inconclusive**.

| strategy | n tasks | 7b - 3b judge fact coverage | status |
| --- | --- | --- | --- |
| full_context | 40 | -0.021 [-0.105, +0.058] | inconclusive |
| rag_topk | 40 | -0.024 [-0.096, +0.047] | inconclusive |

Disagreements are sampled in `judge_disagreements.md`; proposed scorer aliases are review-only in `proposed_fact_aliases.md`.
