# QIRA report — Qwen2.5-7B-Instruct

Protocol `QIRA-v8-transport` · preset `quick` · calibration source `natural` · decision layer 14 / 28

QIRA is backbone-frozen and inference-time adaptive; its K/V transport maps are calibrated offline in closed form on a pool disjoint from all evaluation data.

## 1. The problem: cross-layer K/V mismatch (calibration hold-out)

Mean held-out NMSE of halted-token K/V over all layers after the halting layer (`ridge(sub)`/`mlp` are measured on the MLP subsample).

| policy      |   nmse_kv |
|:------------|----------:|
| stale       |    2.7587 |
| reproject   |    0.3312 |
| mlp         |    0.1472 |
| ridge_layer |    0.1330 |
| ridge(sub)  |    0.1326 |
| ridge       |    0.1326 |
| ridge_plain |    0.1316 |

![](figures/fig1_transport_mismatch.png)

Spearman correlation of each selector score with (a) low halting harm, harm = attention received x ridge transport error summed over layers >= d, and (b) transportability (−log ridge NMSE):

| selector              |   spearman_vs_low_harm |   spearman_vs_transportability |
|:----------------------|-----------------------:|-------------------------------:|
| qei                   |                  0.077 |                         -0.053 |
| qei_equal             |                  0.094 |                         -0.052 |
| qei_no_purity         |                  0.109 |                         -0.050 |
| qei_no_rank           |                  0.099 |                         -0.052 |
| qei_no_gap            |                  0.077 |                         -0.057 |
| qei_no_stability      |                 -0.001 |                         -0.037 |
| qei_exact             |                  0.035 |                         -0.036 |
| cosine                |                  0.146 |                         -0.014 |
| normdelta             |                  0.110 |                         -0.089 |
| drift                 |                 -0.010 |                          0.237 |
| lastattn              |                  0.194 |                          0.026 |
| relnorm               |                 -0.005 |                         -0.197 |
| random                |                  0.009 |                         -0.005 |
| qei_fit               |                  0.138 |                         -0.036 |
| learned               |                  0.467 |                          0.139 |
| feature:purity        |                 -0.004 |                         -0.040 |
| feature:rank_score    |                 -0.004 |                         -0.040 |
| feature:gap           |                  0.023 |                         -0.002 |
| feature:stability     |                  0.134 |                         -0.043 |
| feature:cosine        |                  0.146 |                         -0.014 |
| feature:neg_normdelta |                  0.110 |                         -0.089 |
| feature:neg_drift     |                 -0.010 |                          0.237 |
| feature:neg_relnorm   |                 -0.005 |                         -0.197 |
| feature:neg_lastattn  |                  0.194 |                          0.026 |
| feature:rel_pos       |                  0.408 |                          0.090 |

## 2. Matched token-reduction budget

Dense: 3218 samples; mean Dense prefill 0.171s; engine-vs-HF KL (sanity) nan.

| method      |   budget |    n |   halt_frac |   flop_reduction |   retention |   retention_lo |   retention_hi |   kl_last |   top1_agree |   fidelity |   speedup_long_median |   speedup_latency |
|:------------|---------:|-----:|------------:|-----------------:|------------:|---------------:|---------------:|----------:|-------------:|-----------:|----------------------:|------------------:|
| QIRA        |    0.300 | 3218 |       0.297 |            0.144 |       0.991 |          0.972 |          1.011 |     0.184 |        0.878 |      0.707 |                 1.082 |             1.086 |
| QIRA-L      |    0.300 | 3218 |       0.297 |            0.149 |       0.954 |          0.935 |          0.971 |     0.165 |        0.907 |      0.777 |                 1.084 |             1.085 |
| TokenSparse |    0.300 | 3218 |       0.000 |            0.000 |       0.875 |          0.850 |          0.898 |     0.882 |        0.687 |      0.456 |                 0.909 |             0.921 |
| QIRA        |    0.500 | 3218 |       0.495 |            0.240 |       0.961 |          0.937 |          0.983 |     0.358 |        0.822 |      0.638 |                 1.195 |             1.196 |
| QIRA-L      |    0.500 | 3218 |       0.495 |            0.246 |       0.945 |          0.924 |          0.967 |     0.337 |        0.861 |      0.700 |                 1.198 |             1.199 |
| TokenSparse |    0.500 | 3218 |       0.000 |            0.000 |       0.821 |          0.795 |          0.847 |     1.224 |        0.638 |      0.402 |                 0.968 |             0.969 |
| QIRA        |    0.700 | 3218 |       0.694 |            0.336 |       0.928 |          0.901 |          0.949 |     0.646 |        0.765 |      0.560 |                 1.351 |             1.351 |
| QIRA-L      |    0.700 | 3218 |       0.694 |            0.341 |       0.915 |          0.894 |          0.937 |     0.517 |        0.818 |      0.631 |                 1.352 |             1.354 |
| TokenSparse |    0.700 | 3218 |       0.000 |            0.000 |       0.797 |          0.767 |          0.825 |     1.286 |        0.622 |      0.394 |                 0.977 |             0.976 |
| QIRA        |    0.900 | 3218 |       0.892 |            0.433 |       0.866 |          0.842 |          0.893 |     0.909 |        0.713 |      0.489 |                 1.513 |             1.516 |
| QIRA-L      |    0.900 | 3218 |       0.892 |            0.435 |       0.891 |          0.869 |          0.917 |     0.658 |        0.787 |      0.562 |                 1.501 |             1.505 |
| TokenSparse |    0.900 | 3218 |       0.000 |            0.000 |       0.797 |          0.767 |          0.825 |     1.286 |        0.622 |      0.394 |                 0.977 |             0.976 |

![](figures/fig3_methods.png)

## 3. Matched quality (accuracy-matched efficiency)

Largest efficiency at which each method still retains the target fraction of Dense quality (linear interpolation along the budget sweep; `saturated` = target still met at the largest budget tested, so the value is a lower bound).

| method      |   target |   flop_reduction |   halt_frac |   speedup_long_median |   speedup_latency | reached   | saturated   |
|:------------|---------:|-----------------:|------------:|----------------------:|------------------:|:----------|:------------|
| QIRA        |    0.990 |            0.147 |       0.302 |                 1.085 |             1.089 | True      | False       |
| QIRA        |    0.970 |            0.211 |       0.435 |                 1.160 |             1.163 | True      | False       |
| QIRA        |    0.950 |            0.272 |       0.560 |                 1.246 |             1.247 | True      | False       |
| QIRA        |    0.900 |            0.380 |       0.783 |                 1.424 |             1.425 | True      | False       |
| QIRA        |    0.850 |            0.433 |       0.892 |                 1.513 |             1.516 | True      | True        |
| QIRA-L      |    0.990 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| QIRA-L      |    0.970 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| QIRA-L      |    0.950 |            0.192 |       0.385 |                 1.135 |             1.135 | True      | False       |
| QIRA-L      |    0.900 |            0.401 |       0.819 |                 1.446 |             1.450 | True      | False       |
| QIRA-L      |    0.850 |            0.435 |       0.892 |                 1.501 |             1.505 | True      | True        |
| TokenSparse |    0.990 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| TokenSparse |    0.970 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| TokenSparse |    0.950 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| TokenSparse |    0.900 |            0.000 |       0.000 |                 0.000 |             0.000 | False     | False       |
| TokenSparse |    0.850 |            0.000 |       0.000 |                 0.909 |             0.921 | True      | False       |

## 4. QIRA operating frontier

![](figures/fig2_qira_frontier.png)

_(not run)_

## 4b. Decomposition: plugging QIRA transport into baseline selectors (`+T` = same selector, ridge transport); QIRA vs QIRA-L = QEI vs learned selector

|   budget | base   | variant   |   retention_base |   retention_variant |   kl_base |   kl_variant |
|---------:|:-------|:----------|-----------------:|--------------------:|----------:|-------------:|
|    0.300 | QIRA   | QIRA-L    |            0.991 |               0.954 |     0.184 |        0.165 |
|    0.500 | QIRA   | QIRA-L    |            0.961 |               0.945 |     0.358 |        0.337 |
|    0.700 | QIRA   | QIRA-L    |            0.928 |               0.915 |     0.646 |        0.517 |
|    0.900 | QIRA   | QIRA-L    |            0.866 |               0.891 |     0.909 |        0.658 |

## 4c. Wall-clock prefill latency (LongBench prompts >= 2K tokens; warm-up + median of repeats; no generation)

| method      |   budget |   n |   prompt_tokens |   halt_frac |   flop_reduction |   prefill_s |   prefill_dense |   speedup |   speedup_p25 |   speedup_p75 |   kl_last | attn_mode    |
|:------------|---------:|----:|----------------:|------------:|-----------------:|------------:|----------------:|----------:|--------------:|--------------:|----------:|:-------------|
| QIRA        |    0.300 |   8 |        4053.500 |       0.298 |            0.145 |       0.170 |           0.185 |     1.086 |         1.083 |         1.107 |     0.096 | flex         |
| QIRA        |    0.500 |   8 |        4053.500 |       0.496 |            0.241 |       0.154 |           0.185 |     1.196 |         1.192 |         1.237 |     0.610 | flex         |
| QIRA        |    0.700 |   8 |        4053.500 |       0.695 |            0.337 |       0.137 |           0.185 |     1.351 |         1.347 |         1.371 |     1.601 | flex         |
| QIRA        |    0.900 |   8 |        4053.500 |       0.893 |            0.433 |       0.122 |           0.185 |     1.516 |         1.512 |         1.539 |     3.088 | flex         |
| QIRA-L      |    0.300 |   8 |        4053.500 |       0.298 |            0.150 |       0.170 |           0.185 |     1.085 |         1.082 |         1.105 |     0.275 | flex         |
| QIRA-L      |    0.500 |   8 |        4053.500 |       0.496 |            0.247 |       0.154 |           0.185 |     1.199 |         1.192 |         1.237 |     0.804 | flex         |
| QIRA-L      |    0.700 |   8 |        4053.500 |       0.695 |            0.342 |       0.137 |           0.185 |     1.354 |         1.352 |         1.368 |     1.569 | flex         |
| QIRA-L      |    0.900 |   8 |        4053.500 |       0.893 |            0.436 |       0.123 |           0.185 |     1.505 |         1.495 |         1.526 |     1.812 | flex         |
| TokenSparse |    0.300 |   8 |        4053.500 |       0.000 |            0.000 |       0.203 |           0.185 |     0.921 |         0.910 |         0.934 |     2.657 | token_sparse |
| TokenSparse |    0.500 |   8 |        4053.500 |       0.000 |            0.000 |       0.191 |           0.185 |     0.969 |         0.964 |         0.984 |     3.286 | token_sparse |
| TokenSparse |    0.700 |   8 |        4053.500 |       0.000 |            0.000 |       0.189 |           0.185 |     0.976 |         0.975 |         0.988 |     3.395 | token_sparse |
| TokenSparse |    0.900 |   8 |        4053.500 |       0.000 |            0.000 |       0.189 |           0.185 |     0.976 |         0.972 |         0.988 |     3.395 | token_sparse |

## Metric definitions

* `halt_frac`: halted tokens / prompt tokens (token reduction).
* `token_layer_reduction`: fraction of (token, layer) pairs skipped.
* `flop_reduction`: analytic prefill FLOPs incl. causal attention and transport overhead.
* `speedup_*`: wall-clock prefill time of the Dense engine / method on the same sample (CUDA-synchronised; TTFT = prefill + first-token selection).
* `kl_last`: KL(Dense || method) of the next-token distribution after the prompt.
* `ans_nll_delta`: change in teacher-forced NLL of the reference answer.
* `fidelity`: ROUGE-L between the method's and Dense's generations.
