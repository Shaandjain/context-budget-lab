| strategy | comparison | delta fact coverage | delta citation precision | delta citation recall | delta schema ok | delta abstain correct | delta p50 latency s | delta mean input tokens | comparison run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_context | qwen2.5:7b - qwen2.5:3b | -0.060 | +0.130 | +0.129 | +0.135 | -0.100 | +3.083s | +3.925 | results/local-matrix/qwen2-5-7b/full_context/context-budget-20260612-034806 |
| prefix_cache_friendly | qwen2.5:7b - qwen2.5:3b | +0.003 | +0.079 | +0.200 | +0.165 | +0.100 | +3.112s | +8.725 | results/local-matrix/qwen2-5-7b/prefix_cache_friendly/context-budget-20260612-050805 |
| rag_topk | qwen2.5:7b - qwen2.5:3b | -0.040 | +0.132 | +0.059 | +0.070 | -0.167 | +1.463s | +0.975 | results/local-matrix/qwen2-5-7b/rag_topk/context-budget-20260612-040957 |
| structured_memory | qwen2.5:7b - qwen2.5:3b | +0.017 | +0.079 | -0.065 | -0.175 | +0.321 | +3.567s | -9.000 | results/local-matrix/qwen2-5-7b/structured_memory/context-budget-20260612-044524 |
| summary_memory | qwen2.5:7b - qwen2.5:3b | -0.042 | -0.049 | -0.012 | -0.190 | +0.010 | +2.441s | -0.050 | results/local-matrix/qwen2-5-7b/summary_memory/context-budget-20260612-042607 |
