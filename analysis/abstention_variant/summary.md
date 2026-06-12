# Abstention Variant: synthetic_agent_memory_v0

## Conditions

| model | condition | strategy | n | errors | abstain correct | fact coverage | schema ok | useful answers | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:3b | baseline | prefix_cache_friendly | 100 | 0 | 0.100 | 0.606 | 0.800 | 38 | results/local-matrix/qwen2-5-3b/prefix_cache_friendly/context-budget-20260612-023800 |
| qwen2.5:3b | abstain_variant | prefix_cache_abstain | 100 | 0 | 0.000 | 0.527 | 0.850 | 40 | results/abstention-variant/qwen2-5-3b/prefix_cache_abstain/context-budget-20260612-053230 |
| qwen2.5:7b | baseline | prefix_cache_friendly | 100 | 0 | 0.200 | 0.657 | 0.950 | 48 | results/local-matrix/qwen2-5-7b/prefix_cache_friendly/context-budget-20260612-050805 |
| qwen2.5:7b | abstain_variant | prefix_cache_abstain | 100 | 0 | 0.633 | 0.557 | 0.810 | 25 | results/abstention-variant/qwen2-5-7b/prefix_cache_abstain/context-budget-20260612-053623 |

## Deltas

| model | delta abstain correct | delta fact coverage | delta schema ok | delta useful answers |
| --- | --- | --- | --- | --- |
| qwen2.5:3b | -0.100 | -0.079 | +0.050 | +2 |
| qwen2.5:7b | +0.433 | -0.100 | -0.140 | -23 |
