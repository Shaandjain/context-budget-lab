# A13 v2 Haystack Sweep

Results root: `results/v2-matrix`.
Clean matrix: `True`.

## Completeness

| model | strategy | observed | expected | errors | complete | run |
| --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | full_context | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-3b-instruct/full_context/context-budget-20260703-022226` |
| qwen2.5-3b-instruct | rag_topk | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-3b-instruct/rag_topk/context-budget-20260703-030945` |
| qwen2.5-3b-instruct | structured_memory | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-3b-instruct/structured_memory/context-budget-20260706-175353` |
| qwen2.5-3b-instruct | summary_memory | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-3b-instruct/summary_memory/context-budget-20260706-172537` |
| qwen2.5-7b-instruct | full_context | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-7b-instruct/full_context/context-budget-20260706-182714` |
| qwen2.5-7b-instruct | rag_topk | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-7b-instruct/rag_topk/context-budget-20260706-191451` |
| qwen2.5-7b-instruct | structured_memory | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-7b-instruct/structured_memory/context-budget-20260706-223401` |
| qwen2.5-7b-instruct | summary_memory | 540 | 540 | 0 | True | `results/v2-matrix/qwen2-5-7b-instruct/summary_memory/context-budget-20260706-200602` |

## H1: RAG vs Full Context

| model | hypothesis | strategy | verdict | rationale |
| --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | H1 |  | inconclusive | h32 does not show a positive paired advantage with a CI excluding zero |
| qwen2.5-7b-instruct | H1 |  | inconclusive | h32 does not show a positive paired advantage with a CI excluding zero |

| model | h | comparison | n tasks | fact coverage delta |
| --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | 2 | rag_topk - full_context | 60 | 0.058 [0.016, 0.109] |
| qwen2.5-3b-instruct | 8 | rag_topk - full_context | 60 | 0.039 [-0.010, 0.097] |
| qwen2.5-3b-instruct | 32 | rag_topk - full_context | 60 | 0.029 [-0.026, 0.088] |
| qwen2.5-7b-instruct | 2 | rag_topk - full_context | 60 | -0.006 [-0.029, 0.014] |
| qwen2.5-7b-instruct | 8 | rag_topk - full_context | 60 | -0.014 [-0.066, 0.034] |
| qwen2.5-7b-instruct | 32 | rag_topk - full_context | 60 | 0.011 [-0.035, 0.061] |

## H3: Compression vs RAG

| model | hypothesis | strategy | verdict | rationale |
| --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | H3 | structured_memory | inconclusive | structured_memory does not show a clear h32 degradation relative to rag_topk |
| qwen2.5-3b-instruct | H3 | summary_memory | inconclusive | summary_memory does not show a clear h32 degradation relative to rag_topk |
| qwen2.5-7b-instruct | H3 | structured_memory | supported | structured_memory is farther below rag_topk at h32 than h2, with h32 CI below zero |
| qwen2.5-7b-instruct | H3 | summary_memory | inconclusive | summary_memory does not show a clear h32 degradation relative to rag_topk |

| model | h | comparison | n tasks | fact coverage delta |
| --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | 2 | summary_memory - rag_topk | 60 | -0.030 [-0.121, 0.054] |
| qwen2.5-3b-instruct | 8 | summary_memory - rag_topk | 60 | 0.031 [-0.056, 0.116] |
| qwen2.5-3b-instruct | 32 | summary_memory - rag_topk | 60 | -0.024 [-0.121, 0.067] |
| qwen2.5-3b-instruct | 2 | structured_memory - rag_topk | 60 | -0.121 [-0.188, -0.059] |
| qwen2.5-3b-instruct | 8 | structured_memory - rag_topk | 60 | -0.111 [-0.183, -0.044] |
| qwen2.5-3b-instruct | 32 | structured_memory - rag_topk | 60 | -0.120 [-0.192, -0.054] |
| qwen2.5-7b-instruct | 2 | summary_memory - rag_topk | 60 | 0.039 [-0.042, 0.114] |
| qwen2.5-7b-instruct | 8 | summary_memory - rag_topk | 60 | 0.049 [-0.028, 0.124] |
| qwen2.5-7b-instruct | 32 | summary_memory - rag_topk | 60 | 0.017 [-0.073, 0.098] |
| qwen2.5-7b-instruct | 2 | structured_memory - rag_topk | 60 | -0.082 [-0.149, -0.024] |
| qwen2.5-7b-instruct | 8 | structured_memory - rag_topk | 60 | -0.071 [-0.137, -0.009] |
| qwen2.5-7b-instruct | 32 | structured_memory - rag_topk | 60 | -0.090 [-0.152, -0.033] |

## Frontier

| model | h | strategy | fact coverage | p50 latency s | mean input tokens | Pareto optimal |
| --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-3b-instruct | 2 | full_context | 0.556 | 4.663 | 244.683 | False |
| qwen2.5-3b-instruct | 2 | rag_topk | 0.614 | 4.213 | 246.333 | True |
| qwen2.5-3b-instruct | 2 | structured_memory | 0.493 | 2.818 | 207.817 | False |
| qwen2.5-3b-instruct | 2 | summary_memory | 0.584 | 2.640 | 195.467 | True |
| qwen2.5-3b-instruct | 8 | full_context | 0.544 | 4.696 | 607.383 | False |
| qwen2.5-3b-instruct | 8 | rag_topk | 0.583 | 4.498 | 305.533 | True |
| qwen2.5-3b-instruct | 8 | structured_memory | 0.472 | 3.264 | 474.067 | False |
| qwen2.5-3b-instruct | 8 | summary_memory | 0.614 | 3.117 | 428.783 | True |
| qwen2.5-3b-instruct | 32 | full_context | 0.563 | 5.088 | 2145.033 | False |
| qwen2.5-3b-instruct | 32 | rag_topk | 0.591 | 4.435 | 312.383 | True |
| qwen2.5-3b-instruct | 32 | structured_memory | 0.471 | 3.152 | 1568.900 | False |
| qwen2.5-3b-instruct | 32 | summary_memory | 0.568 | 2.737 | 1391.717 | True |
| qwen2.5-7b-instruct | 2 | full_context | 0.573 | 4.510 | 244.683 | True |
| qwen2.5-7b-instruct | 2 | rag_topk | 0.567 | 4.895 | 246.333 | False |
| qwen2.5-7b-instruct | 2 | structured_memory | 0.485 | 4.824 | 207.817 | True |
| qwen2.5-7b-instruct | 2 | summary_memory | 0.606 | 5.027 | 195.467 | True |
| qwen2.5-7b-instruct | 8 | full_context | 0.586 | 4.529 | 607.383 | True |
| qwen2.5-7b-instruct | 8 | rag_topk | 0.572 | 4.786 | 305.533 | True |
| qwen2.5-7b-instruct | 8 | structured_memory | 0.500 | 4.569 | 474.067 | True |
| qwen2.5-7b-instruct | 8 | summary_memory | 0.620 | 5.049 | 428.783 | True |
| qwen2.5-7b-instruct | 32 | full_context | 0.578 | 4.655 | 2145.033 | True |
| qwen2.5-7b-instruct | 32 | rag_topk | 0.589 | 5.168 | 312.383 | True |
| qwen2.5-7b-instruct | 32 | structured_memory | 0.499 | 5.053 | 1568.900 | True |
| qwen2.5-7b-instruct | 32 | summary_memory | 0.606 | 5.224 | 1391.717 | True |
