# Paired Delta Reanalysis

Deltas are comparison minus baseline. Each arm is first averaged per task across repeats, then bootstrapped over task-level deltas.

## Paired-Only Separations

| type | baseline | comparison | metric | n tasks | paired delta |
| --- | --- | --- | --- | --- | --- |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b summary_memory | fact_coverage | 40 | -0.148 [-0.246, -0.053] |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b structured_memory | abstain_correct | 6 | -0.500 [-0.900, -0.033] |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b summary_memory | fact_coverage | 40 | -0.130 [-0.225, -0.041] |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b structured_memory | fact_coverage | 40 | -0.190 [-0.287, -0.098] |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | -0.229 [-0.382, -0.059] |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b structured_memory | fact_coverage | 40 | -0.060 [-0.114, -0.007] |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.094 [0.002, 0.194] |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.154 [0.059, 0.244] |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | -0.218 [-0.376, -0.059] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b summary_memory | fact_coverage | 40 | -0.130 [-0.219, -0.044] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | fact_coverage | 40 | -0.132 [-0.219, -0.038] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | citation_precision | 35 | -0.128 [-0.267, -0.012] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | citation_recall | 34 | -0.141 [-0.271, -0.029] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | -0.094 [-0.194, -0.012] |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | -0.533 [-0.867, -0.233] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | fact_coverage | 40 | -0.132 [-0.227, -0.037] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | citation_precision | 38 | -0.230 [-0.348, -0.120] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | citation_recall | 34 | -0.141 [-0.265, -0.018] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | fact_coverage | 40 | -0.134 [-0.217, -0.047] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | citation_precision | 36 | -0.140 [-0.269, -0.008] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | citation_recall | 34 | -0.135 [-0.276, -0.006] |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | -0.088 [-0.182, -0.012] |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.138 [0.040, 0.236] |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | citation_precision | 38 | 0.208 [0.088, 0.314] |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | -0.467 [-0.833, -0.133] |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.140 [0.045, 0.237] |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | citation_precision | 36 | 0.117 [0.003, 0.232] |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | -0.400 [-0.767, -0.100] |
| model_pair | qwen2.5:3b full_context | qwen2.5:7b full_context | citation_recall | 34 | 0.129 [0.029, 0.247] |
| model_pair | qwen2.5:3b rag_topk | qwen2.5:7b rag_topk | citation_precision | 38 | 0.095 [0.013, 0.177] |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | citation_precision | 36 | 0.157 [0.020, 0.299] |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.200 [0.059, 0.347] |

## All Comparisons

| type | baseline | comparison | metric | n tasks | baseline | comparison | paired delta | per-arm overlap | paired separates | paired-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b rag_topk | fact_coverage | 40 | 0.785 [0.678, 0.876] | 0.767 [0.670, 0.854] | -0.018 [-0.086, 0.040] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b rag_topk | citation_precision | 40 | 0.619 [0.492, 0.738] | 0.606 [0.490, 0.729] | -0.013 [-0.095, 0.073] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b rag_topk | citation_recall | 34 | 0.847 [0.729, 0.947] | 0.912 [0.818, 0.982] | 0.065 [-0.029, 0.171] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b rag_topk | abstain_correct | 6 | 0.833 [0.500, 1.000] | 0.667 [0.333, 1.000] | -0.167 [-0.500, 0.000] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b summary_memory | fact_coverage | 40 | 0.785 [0.685, 0.873] | 0.636 [0.554, 0.718] | -0.148 [-0.246, -0.053] | True | True | True |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b summary_memory | citation_precision | 39 | 0.635 [0.509, 0.757] | 0.568 [0.443, 0.698] | -0.067 [-0.215, 0.079] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b summary_memory | citation_recall | 34 | 0.847 [0.724, 0.947] | 0.841 [0.729, 0.941] | -0.006 [-0.165, 0.141] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b summary_memory | abstain_correct | 6 | 0.833 [0.500, 1.000] | 0.767 [0.500, 0.967] | -0.067 [-0.233, 0.067] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b structured_memory | fact_coverage | 40 | 0.785 [0.681, 0.881] | 0.576 [0.483, 0.668] | -0.208 [-0.300, -0.115] | False | True | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b structured_memory | citation_precision | 40 | 0.619 [0.492, 0.740] | 0.555 [0.437, 0.670] | -0.064 [-0.200, 0.079] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b structured_memory | citation_recall | 34 | 0.847 [0.712, 0.947] | 0.900 [0.806, 0.971] | 0.053 [-0.071, 0.176] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b structured_memory | abstain_correct | 6 | 0.833 [0.500, 1.000] | 0.333 [0.067, 0.633] | -0.500 [-0.900, -0.033] | True | True | True |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.785 [0.680, 0.872] | 0.730 [0.633, 0.822] | -0.054 [-0.123, 0.006] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b prefix_cache_friendly | citation_precision | 36 | 0.688 [0.566, 0.815] | 0.593 [0.438, 0.739] | -0.095 [-0.265, 0.078] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | 0.847 [0.724, 0.941] | 0.682 [0.535, 0.824] | -0.165 [-0.347, 0.012] | True | False | False |
| strategy_pair | qwen2.5:3b full_context | qwen2.5:3b prefix_cache_friendly | abstain_correct | 6 | 0.833 [0.500, 1.000] | 0.100 [0.000, 0.300] | -0.733 [-1.000, -0.400] | False | True | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b summary_memory | fact_coverage | 40 | 0.767 [0.670, 0.857] | 0.636 [0.553, 0.728] | -0.130 [-0.225, -0.041] | True | True | True |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b summary_memory | citation_precision | 39 | 0.621 [0.503, 0.733] | 0.568 [0.445, 0.687] | -0.053 [-0.200, 0.100] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b summary_memory | citation_recall | 34 | 0.912 [0.812, 0.988] | 0.841 [0.712, 0.947] | -0.071 [-0.224, 0.082] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b summary_memory | abstain_correct | 6 | 0.667 [0.333, 1.000] | 0.767 [0.500, 0.967] | 0.100 [-0.200, 0.467] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b structured_memory | fact_coverage | 40 | 0.767 [0.665, 0.856] | 0.576 [0.492, 0.669] | -0.190 [-0.287, -0.098] | True | True | True |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b structured_memory | citation_precision | 40 | 0.606 [0.491, 0.727] | 0.555 [0.429, 0.667] | -0.051 [-0.157, 0.053] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b structured_memory | citation_recall | 34 | 0.912 [0.824, 0.988] | 0.900 [0.812, 0.971] | -0.012 [-0.094, 0.076] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b structured_memory | abstain_correct | 6 | 0.667 [0.333, 1.000] | 0.333 [0.067, 0.667] | -0.333 [-0.733, 0.067] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.767 [0.663, 0.853] | 0.730 [0.626, 0.817] | -0.036 [-0.089, 0.014] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b prefix_cache_friendly | citation_precision | 36 | 0.673 [0.562, 0.776] | 0.593 [0.443, 0.739] | -0.080 [-0.226, 0.075] | True | False | False |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | 0.912 [0.818, 0.988] | 0.682 [0.529, 0.824] | -0.229 [-0.382, -0.059] | True | True | True |
| strategy_pair | qwen2.5:3b rag_topk | qwen2.5:3b prefix_cache_friendly | abstain_correct | 6 | 0.667 [0.333, 1.000] | 0.100 [0.000, 0.300] | -0.567 [-0.900, -0.233] | False | True | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b structured_memory | fact_coverage | 40 | 0.636 [0.547, 0.728] | 0.576 [0.484, 0.678] | -0.060 [-0.114, -0.007] | True | True | True |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b structured_memory | citation_precision | 39 | 0.568 [0.442, 0.691] | 0.569 [0.448, 0.686] | 0.001 [-0.130, 0.145] | True | False | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b structured_memory | citation_recall | 34 | 0.841 [0.724, 0.947] | 0.900 [0.806, 0.971] | 0.059 [-0.071, 0.188] | True | False | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b structured_memory | abstain_correct | 6 | 0.767 [0.500, 0.967] | 0.333 [0.067, 0.667] | -0.433 [-0.867, 0.033] | True | False | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.636 [0.552, 0.719] | 0.730 [0.633, 0.821] | 0.094 [0.002, 0.194] | True | True | True |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b prefix_cache_friendly | citation_precision | 36 | 0.616 [0.493, 0.740] | 0.593 [0.433, 0.737] | -0.023 [-0.197, 0.160] | True | False | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | 0.841 [0.712, 0.947] | 0.682 [0.524, 0.824] | -0.159 [-0.359, 0.035] | True | False | False |
| strategy_pair | qwen2.5:3b summary_memory | qwen2.5:3b prefix_cache_friendly | abstain_correct | 6 | 0.767 [0.533, 0.967] | 0.100 [0.000, 0.300] | -0.667 [-0.933, -0.400] | False | True | False |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | fact_coverage | 40 | 0.576 [0.481, 0.665] | 0.730 [0.635, 0.820] | 0.154 [0.059, 0.244] | True | True | True |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | citation_precision | 36 | 0.617 [0.505, 0.729] | 0.593 [0.440, 0.741] | -0.024 [-0.197, 0.146] | True | False | False |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | citation_recall | 34 | 0.900 [0.812, 0.971] | 0.682 [0.529, 0.824] | -0.218 [-0.376, -0.059] | True | True | True |
| strategy_pair | qwen2.5:3b structured_memory | qwen2.5:3b prefix_cache_friendly | abstain_correct | 6 | 0.333 [0.067, 0.667] | 0.100 [0.000, 0.300] | -0.233 [-0.667, 0.200] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b rag_topk | fact_coverage | 40 | 0.725 [0.620, 0.821] | 0.727 [0.620, 0.825] | 0.002 [-0.045, 0.053] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b rag_topk | citation_precision | 36 | 0.757 [0.656, 0.846] | 0.773 [0.667, 0.875] | 0.016 [-0.071, 0.106] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b rag_topk | citation_recall | 34 | 0.976 [0.935, 1.000] | 0.971 [0.912, 1.000] | -0.006 [-0.071, 0.053] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b rag_topk | abstain_correct | 6 | 0.733 [0.400, 0.967] | 0.500 [0.167, 0.833] | -0.233 [-0.500, 0.000] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b summary_memory | fact_coverage | 40 | 0.725 [0.620, 0.823] | 0.595 [0.508, 0.684] | -0.130 [-0.219, -0.044] | True | True | True |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b summary_memory | citation_precision | 36 | 0.757 [0.656, 0.847] | 0.531 [0.426, 0.633] | -0.227 [-0.348, -0.101] | False | True | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b summary_memory | citation_recall | 34 | 0.976 [0.929, 1.000] | 0.829 [0.735, 0.924] | -0.147 [-0.259, -0.035] | False | True | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b summary_memory | abstain_correct | 6 | 0.733 [0.400, 0.967] | 0.667 [0.333, 1.000] | -0.067 [-0.400, 0.200] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | fact_coverage | 40 | 0.725 [0.611, 0.816] | 0.593 [0.497, 0.688] | -0.132 [-0.219, -0.038] | True | True | True |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | citation_precision | 35 | 0.779 [0.685, 0.868] | 0.651 [0.528, 0.771] | -0.128 [-0.267, -0.012] | True | True | True |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | citation_recall | 34 | 0.976 [0.929, 1.000] | 0.835 [0.712, 0.941] | -0.141 [-0.271, -0.029] | True | True | True |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b structured_memory | abstain_correct | 6 | 0.733 [0.400, 0.967] | 0.600 [0.200, 0.933] | -0.133 [-0.400, 0.000] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.725 [0.626, 0.818] | 0.733 [0.637, 0.818] | 0.008 [-0.073, 0.086] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | citation_precision | 37 | 0.737 [0.628, 0.842] | 0.730 [0.603, 0.847] | -0.007 [-0.130, 0.097] | True | False | False |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.976 [0.929, 1.000] | 0.882 [0.771, 0.971] | -0.094 [-0.194, -0.012] | True | True | True |
| strategy_pair | qwen2.5:7b full_context | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | 0.733 [0.400, 0.967] | 0.200 [0.000, 0.533] | -0.533 [-0.867, -0.233] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | fact_coverage | 40 | 0.727 [0.623, 0.821] | 0.595 [0.507, 0.679] | -0.132 [-0.227, -0.037] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | citation_precision | 38 | 0.732 [0.617, 0.849] | 0.503 [0.396, 0.618] | -0.230 [-0.348, -0.120] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | citation_recall | 34 | 0.971 [0.912, 1.000] | 0.829 [0.735, 0.918] | -0.141 [-0.265, -0.018] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b summary_memory | abstain_correct | 6 | 0.500 [0.167, 0.833] | 0.667 [0.167, 1.000] | 0.167 [0.000, 0.500] | True | False | False |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | fact_coverage | 40 | 0.727 [0.625, 0.822] | 0.593 [0.499, 0.694] | -0.134 [-0.217, -0.047] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | citation_precision | 36 | 0.773 [0.659, 0.867] | 0.633 [0.511, 0.748] | -0.140 [-0.269, -0.008] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | citation_recall | 34 | 0.971 [0.912, 1.000] | 0.835 [0.712, 0.941] | -0.135 [-0.276, -0.006] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b structured_memory | abstain_correct | 6 | 0.500 [0.167, 0.833] | 0.600 [0.200, 0.933] | 0.100 [0.000, 0.300] | True | False | False |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.727 [0.626, 0.826] | 0.733 [0.646, 0.815] | 0.006 [-0.069, 0.077] | True | False | False |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b prefix_cache_friendly | citation_precision | 38 | 0.732 [0.615, 0.836] | 0.711 [0.582, 0.832] | -0.022 [-0.129, 0.082] | True | False | False |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.971 [0.912, 1.000] | 0.882 [0.782, 0.971] | -0.088 [-0.182, -0.012] | True | True | True |
| strategy_pair | qwen2.5:7b rag_topk | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | 0.500 [0.167, 0.833] | 0.200 [0.000, 0.533] | -0.300 [-0.633, 0.000] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b structured_memory | fact_coverage | 40 | 0.595 [0.511, 0.680] | 0.593 [0.500, 0.685] | -0.002 [-0.071, 0.061] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b structured_memory | citation_precision | 36 | 0.531 [0.426, 0.632] | 0.633 [0.514, 0.754] | 0.102 [-0.013, 0.221] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b structured_memory | citation_recall | 34 | 0.829 [0.729, 0.918] | 0.835 [0.712, 0.941] | 0.006 [-0.124, 0.129] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b structured_memory | abstain_correct | 6 | 0.667 [0.167, 1.000] | 0.600 [0.200, 0.933] | -0.067 [-0.200, 0.000] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.595 [0.507, 0.680] | 0.733 [0.639, 0.817] | 0.138 [0.040, 0.236] | True | True | True |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | citation_precision | 38 | 0.503 [0.389, 0.612] | 0.711 [0.590, 0.825] | 0.208 [0.088, 0.314] | True | True | True |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.829 [0.718, 0.918] | 0.882 [0.771, 0.971] | 0.053 [-0.076, 0.200] | True | False | False |
| strategy_pair | qwen2.5:7b summary_memory | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | 0.667 [0.167, 1.000] | 0.200 [0.000, 0.533] | -0.467 [-0.833, -0.133] | True | True | True |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.593 [0.499, 0.682] | 0.733 [0.636, 0.816] | 0.140 [0.045, 0.237] | True | True | True |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | citation_precision | 36 | 0.633 [0.513, 0.743] | 0.750 [0.619, 0.867] | 0.117 [0.003, 0.232] | True | True | True |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.835 [0.712, 0.941] | 0.882 [0.776, 0.971] | 0.047 [-0.100, 0.194] | True | False | False |
| strategy_pair | qwen2.5:7b structured_memory | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | 0.600 [0.267, 0.933] | 0.200 [0.000, 0.533] | -0.400 [-0.767, -0.100] | True | True | True |
| model_pair | qwen2.5:3b full_context | qwen2.5:7b full_context | fact_coverage | 40 | 0.785 [0.675, 0.874] | 0.725 [0.618, 0.820] | -0.060 [-0.131, 0.001] | True | False | False |
| model_pair | qwen2.5:3b full_context | qwen2.5:7b full_context | citation_precision | 37 | 0.669 [0.541, 0.792] | 0.737 [0.628, 0.832] | 0.068 [-0.031, 0.171] | True | False | False |
| model_pair | qwen2.5:3b full_context | qwen2.5:7b full_context | citation_recall | 34 | 0.847 [0.729, 0.941] | 0.976 [0.935, 1.000] | 0.129 [0.029, 0.247] | True | True | True |
| model_pair | qwen2.5:3b full_context | qwen2.5:7b full_context | abstain_correct | 6 | 0.833 [0.500, 1.000] | 0.733 [0.433, 0.967] | -0.100 [-0.233, 0.000] | True | False | False |
| model_pair | qwen2.5:3b rag_topk | qwen2.5:7b rag_topk | fact_coverage | 40 | 0.767 [0.670, 0.854] | 0.727 [0.629, 0.820] | -0.040 [-0.100, 0.018] | True | False | False |
| model_pair | qwen2.5:3b rag_topk | qwen2.5:7b rag_topk | citation_precision | 38 | 0.638 [0.516, 0.750] | 0.732 [0.624, 0.839] | 0.095 [0.013, 0.177] | True | True | True |
| model_pair | qwen2.5:3b rag_topk | qwen2.5:7b rag_topk | citation_recall | 34 | 0.912 [0.829, 0.988] | 0.971 [0.912, 1.000] | 0.059 [-0.059, 0.176] | True | False | False |
| model_pair | qwen2.5:3b rag_topk | qwen2.5:7b rag_topk | abstain_correct | 6 | 0.667 [0.333, 1.000] | 0.500 [0.167, 0.833] | -0.167 [-0.667, 0.500] | True | False | False |
| model_pair | qwen2.5:3b summary_memory | qwen2.5:7b summary_memory | fact_coverage | 40 | 0.636 [0.546, 0.724] | 0.595 [0.509, 0.680] | -0.042 [-0.107, 0.018] | True | False | False |
| model_pair | qwen2.5:3b summary_memory | qwen2.5:7b summary_memory | citation_precision | 37 | 0.599 [0.465, 0.719] | 0.516 [0.401, 0.626] | -0.083 [-0.220, 0.062] | True | False | False |
| model_pair | qwen2.5:3b summary_memory | qwen2.5:7b summary_memory | citation_recall | 34 | 0.841 [0.712, 0.947] | 0.829 [0.724, 0.918] | -0.012 [-0.153, 0.135] | True | False | False |
| model_pair | qwen2.5:3b summary_memory | qwen2.5:7b summary_memory | abstain_correct | 6 | 0.767 [0.500, 0.967] | 0.667 [0.333, 1.000] | -0.100 [-0.333, 0.067] | True | False | False |
| model_pair | qwen2.5:3b structured_memory | qwen2.5:7b structured_memory | fact_coverage | 40 | 0.576 [0.483, 0.665] | 0.593 [0.496, 0.685] | 0.017 [-0.037, 0.062] | True | False | False |
| model_pair | qwen2.5:3b structured_memory | qwen2.5:7b structured_memory | citation_precision | 36 | 0.617 [0.495, 0.733] | 0.633 [0.501, 0.765] | 0.016 [-0.103, 0.133] | True | False | False |
| model_pair | qwen2.5:3b structured_memory | qwen2.5:7b structured_memory | citation_recall | 34 | 0.900 [0.812, 0.971] | 0.835 [0.718, 0.941] | -0.065 [-0.194, 0.065] | True | False | False |
| model_pair | qwen2.5:3b structured_memory | qwen2.5:7b structured_memory | abstain_correct | 6 | 0.333 [0.067, 0.633] | 0.600 [0.267, 0.933] | 0.267 [-0.333, 0.833] | True | False | False |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | fact_coverage | 40 | 0.730 [0.630, 0.821] | 0.733 [0.649, 0.809] | 0.003 [-0.066, 0.065] | True | False | False |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | citation_precision | 36 | 0.593 [0.439, 0.731] | 0.750 [0.619, 0.867] | 0.157 [0.020, 0.299] | True | True | True |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | citation_recall | 34 | 0.682 [0.529, 0.824] | 0.882 [0.776, 0.971] | 0.200 [0.059, 0.347] | True | True | True |
| model_pair | qwen2.5:3b prefix_cache_friendly | qwen2.5:7b prefix_cache_friendly | abstain_correct | 6 | 0.100 [0.000, 0.300] | 0.200 [0.000, 0.500] | 0.100 [-0.267, 0.533] | True | False | False |
