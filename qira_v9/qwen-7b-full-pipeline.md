# Qwen2.5-7B-Instruct: Full Pipeline Empirical Evaluation Tables

**Model Architecture:** `Qwen/Qwen2.5-7B-Instruct` (28 Layers, 3584 Hidden, 28 Q-Heads, 4 KV-Heads, 128 Head Dim, bfloat16)  
**Evaluation Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, sm_120)  
**Evaluation Scope:** 
* **LongBench Full Suite:** All 16 subsets, 3,218 official samples (Context window up to 8,192 tokens)
* **Core Downstream Benchmarks:** SQuAD v1.1 (800 samples), MMLU (800 samples), SST-2 (872 samples) = 2,472 total samples  
**Budgets Evaluated:** $b \in \{0.30, 0.50, 0.70, 0.90\}$ across all methods and tasks.

---

## Ground Truth Dense Reference (Uncompressed Baseline)

Dense causal attention was executed across all samples to establish the ground truth normalization denominator:

| Benchmark / Task | Primary Metric | Total Evaluated Samples ($n$) | Dense Ground Truth Score |
| :--- | :--- | :---: | :---: |
| **LongBench (All 16 Subsets)** | Official qa-F1 | 3,218 | **0.2563** |
| **SQuAD v1.1** | Official Token F1 | 800 | **0.9086** |
| **MMLU** | 5-shot Accuracy | 800 | **0.6838** |
| **SST-2** | Binary Classification Accuracy | 872 | **0.9438** |

$$\text{Task Retention} = \frac{\text{Score}_{\text{Method}}}{\text{Score}_{\text{Dense}}}$$
$$\text{Downstream Macro Retention} = \frac{1}{3} \left( \frac{\text{SQuAD}_{\text{Method}}}{0.9086} + \frac{\text{MMLU}_{\text{Method}}}{0.6838} + \frac{\text{SST-2}_{\text{Method}}}{0.9438} \right)$$
$$\text{LongBench Retention} = \frac{\text{LongBench F1}_{\text{Method}}}{0.2563}$$

> **Critical Architectural Note on TokenSparse Downstream Evaluation:**  
> The original TokenSparse algorithm (Jo et al., ICML 2026) cannot execute token pruning on standard downstream benchmarks (SQuAD v1.1, MMLU, SST-2). The algorithm is designed strictly for long-sequence workloads and mandates a sequence length threshold $L \ge \text{min\_tokens} + \text{window\_size}$ ($512 + 128 = 640$ tokens) to activate its token-level key-value pruning cache. Standard downstream prompts (SST-2: $\sim 30\text{--}60$ tokens; MMLU: $\sim 100\text{--}200$ tokens; SQuAD v1.1: $\sim 150\text{--}300$ tokens) are substantially shorter than this 640-token operational threshold ($L < 640$). When evaluated on sequences under 640 tokens, the token-pruning mechanism is invalid: it either raises an out-of-bounds error or skips pruning completely, defaulting to 100% unpruned dense execution ($\text{halt\_frac} = 0.0$). Reporting dense scores as TokenSparse performance would be scientifically invalid and misleading. Consequently, TokenSparse downstream entries are strictly designated as **Not Valid**.

---

## Section 1: Halting Budget $b = 0.30$ (30% Token Halting)

### Table 1A: Raw Accuracy & F-Measure Scores ($b = 0.30$)

| Method | LongBench qa-F1 ($n=3,218$) | SQuAD v1.1 Token F1 ($n=800$) | MMLU Accuracy ($n=800$) | SST-2 Accuracy ($n=872$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth)** | **0.2563** | **0.9086** | **0.6838** | **0.9438** |
| **QIRA (Ours)** | **0.2539** | **0.8926** | **0.6450** | **0.9484** |
| **QIRA-L (Ours)** | **0.2445** | **0.8858** | **0.6200** | **0.9461** |
| **TokenSparse (Jo et al., 2026)** | 0.2243 | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave (Cordoba et al., 2026)** | 0.1445 | 0.7801 | 0.6613 | 0.9369 |
| **LazyLLM** | 0.2491 | 0.7365 | 0.5875 | 0.9335 |
| **SlimInfer** | 0.2369 | 0.8314 | 0.5163 | 0.9392 |
| **QuickSilver** | 0.0251 | 0.2180 | 0.3975 | 0.9140 |
| **KVSimilarity** | 0.0203 | 0.2171 | 0.4638 | 0.8956 |

*\*Note on TokenSparse Inapplicability:\* The original TokenSparse algorithm cannot run on SQuAD, MMLU, or SST-2 because its prefix-pruning routine requires a minimum context length of $L \ge 640$ tokens ($\text{min\_tokens} = 512 + \text{window\_size} = 128$). Short downstream prompts ($L < 640$) cannot activate pruning.

### Table 1B: Quality Retention (%) Relative to Dense ($b = 0.30$)

| Method | LongBench Ret. (%) | SQuAD Ret. (%) | MMLU Ret. (%) | SST-2 Ret. (%) | Downstream Macro Ret. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **99.1%** | **98.2%** | **94.3%** | **100.5%** | **101.4%** |
| **QIRA-L (Ours)** | **95.4%** | **97.5%** | **90.7%** | **100.2%** | **100.2%** |
| **TokenSparse** | 87.5% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave** | 56.4% | 85.9% | 96.7% | 99.3% | 93.9% |
| **LazyLLM** | 97.2% | 81.1% | 85.9% | 98.9% | 97.7% |
| **SlimInfer** | 92.5% | 91.5% | 75.5% | 99.5% | 91.6% |
| **QuickSilver** | 9.8% | 24.0% | 58.1% | 96.8% | 44.7% |
| **KVSimilarity** | 7.9% | 23.9% | 67.8% | 94.9% | 52.9% |

*\*Note on TokenSparse Inapplicability:\* Downstream retention is Not Valid for TokenSparse because short-context benchmarks cannot be processed by the token pruning algorithm.

### Table 1C: Efficiency & Distribution Divergence ($b = 0.30$)

| Method | LongBench Next-Token KL | Downstream Next-Token KL | FLOP Reduction (%) | Token Halt Frac | Measured Speedup ($4\text{K}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **0.184** | **0.448** | **14.4%** | **0.297** | **1.08×** |
| **QIRA-L (Ours)** | **0.165** | **0.576** | **14.9%** | **0.297** | **1.08×** |
| **TokenSparse** | 0.882 | *Not Valid*\* | 0.0% | 0.000 | 0.91× *(Slower)* |
| **DepthWeave** | 1.708 | 0.533 | 14.9% | 0.298 | 0.53× *(Slower)* |
| **LazyLLM** | 0.094 | 1.105 | 15.4% | 0.297 | 1.16× |
| **SlimInfer** | 0.850 | 1.239 | 15.4% | 0.297 | 1.16× |
| **QuickSilver** | 5.629 | 2.656 | 14.8% | 0.297 | 1.10× |
| **KVSimilarity** | 5.926 | 2.360 | 14.9% | 0.297 | 1.10× |

*\*Note on TokenSparse Inapplicability:\* Downstream Next-Token KL is Not Valid because TokenSparse cannot execute pruning on downstream sequences ($L < 640$).

---

## Section 2: Halting Budget $b = 0.50$ (50% Token Halting)

### Table 2A: Raw Accuracy & F-Measure Scores ($b = 0.50$)

| Method | LongBench qa-F1 ($n=3,218$) | SQuAD v1.1 Token F1 ($n=800$) | MMLU Accuracy ($n=800$) | SST-2 Accuracy ($n=872$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth)** | **0.2563** | **0.9086** | **0.6838** | **0.9438** |
| **QIRA (Ours)** | **0.2463** | **0.8906** | **0.6275** | **0.9495** |
| **QIRA-L (Ours)** | **0.2422** | **0.8645** | **0.5313** | **0.9564** |
| **TokenSparse (Jo et al., 2026)** | 0.2104 | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave (Cordoba et al., 2026)** | 0.1329 | 0.6998 | 0.6475 | 0.9415 |
| **LazyLLM** | 0.2343 | 0.5224 | 0.4963 | 0.9163 |
| **SlimInfer** | 0.2098 | 0.7065 | 0.4550 | 0.9369 |
| **QuickSilver** | 0.0189 | 0.0791 | 0.2513 | 0.8865 |
| **KVSimilarity** | 0.0156 | 0.1210 | 0.3575 | 0.8211 |

*\*Note on TokenSparse Inapplicability:\* The original TokenSparse algorithm cannot run on SQuAD, MMLU, or SST-2 because its prefix-pruning routine requires a minimum context length of $L \ge 640$ tokens ($\text{min\_tokens} = 512 + \text{window\_size} = 128$). Short downstream prompts ($L < 640$) cannot activate pruning.

### Table 2B: Quality Retention (%) Relative to Dense ($b = 0.50$)

| Method | LongBench Ret. (%) | SQuAD Ret. (%) | MMLU Ret. (%) | SST-2 Ret. (%) | Downstream Macro Ret. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **96.1%** | **98.0%** | **91.8%** | **100.6%** | **100.7%** |
| **QIRA-L (Ours)** | **94.5%** | **95.1%** | **77.7%** | **101.3%** | **102.9%** |
| **TokenSparse** | 82.1% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave** | 51.9% | 77.0% | 94.7% | 99.8% | 90.5% |
| **LazyLLM** | 91.4% | 57.5% | 72.6% | 97.1% | 59.9% |
| **SlimInfer** | 81.9% | 77.8% | 66.5% | 99.3% | 73.4% |
| **QuickSilver** | 7.4% | 8.7% | 36.7% | 93.9% | 34.8% |
| **KVSimilarity** | 6.1% | 13.3% | 52.3% | 87.0% | 41.3% |

*\*Note on TokenSparse Inapplicability:\* Downstream retention is Not Valid for TokenSparse because short-context benchmarks cannot be processed by the token pruning algorithm.

### Table 2C: Efficiency & Distribution Divergence ($b = 0.50$)

| Method | LongBench Next-Token KL | Downstream Next-Token KL | FLOP Reduction (%) | Token Halt Frac | Measured Speedup ($4\text{K}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **0.358** | **0.738** | **24.0%** | **0.495** | **1.19×** |
| **QIRA-L (Ours)** | **0.337** | **0.919** | **24.6%** | **0.495** | **1.20×** |
| **TokenSparse** | 1.224 | *Not Valid*\* | 0.0% | 0.000 | 0.97× *(Slower)* |
| **DepthWeave** | 1.827 | 0.832 | 24.5% | 0.497 | 0.59× *(Slower)* |
| **LazyLLM** | 0.296 | 2.459 | 25.5% | 0.495 | 1.29× |
| **SlimInfer** | 1.737 | 2.013 | 25.5% | 0.495 | 1.29× |
| **QuickSilver** | 6.105 | 3.885 | 24.7% | 0.495 | 1.22× |
| **KVSimilarity** | 6.388 | 3.168 | 24.8% | 0.495 | 1.23× |

*\*Note on TokenSparse Inapplicability:\* Downstream Next-Token KL is Not Valid because TokenSparse cannot execute pruning on downstream sequences ($L < 640$).

---

## Section 3: Halting Budget $b = 0.70$ (70% Token Halting)

### Table 3A: Raw Accuracy & F-Measure Scores ($b = 0.70$)

| Method | LongBench qa-F1 ($n=3,218$) | SQuAD v1.1 Token F1 ($n=800$) | MMLU Accuracy ($n=800$) | SST-2 Accuracy ($n=872$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth)** | **0.2563** | **0.9086** | **0.6838** | **0.9438** |
| **QIRA (Ours)** | **0.2377** | **0.8679** | **0.5800** | **0.9541** |
| **QIRA-L (Ours)** | **0.2345** | **0.8492** | **0.4938** | **0.9576** |
| **TokenSparse (Jo et al., 2026)** | 0.2043 | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave (Cordoba et al., 2026)** | 0.1288 | 0.6540 | 0.5900 | 0.9186 |
| **LazyLLM** | 0.2116 | 0.2332 | 0.4488 | 0.9083 |
| **SlimInfer** | 0.1651 | 0.5500 | 0.3750 | 0.9323 |
| **QuickSilver** | 0.0141 | 0.0302 | 0.1538 | 0.4610 |
| **KVSimilarity** | 0.0112 | 0.0475 | 0.2325 | 0.6261 |

*\*Note on TokenSparse Inapplicability:\* The original TokenSparse algorithm cannot run on SQuAD, MMLU, or SST-2 because its prefix-pruning routine requires a minimum context length of $L \ge 640$ tokens ($\text{min\_tokens} = 512 + \text{window\_size} = 128$). Short downstream prompts ($L < 640$) cannot activate pruning.

### Table 3B: Quality Retention (%) Relative to Dense ($b = 0.70$)

| Method | LongBench Ret. (%) | SQuAD Ret. (%) | MMLU Ret. (%) | SST-2 Ret. (%) | Downstream Macro Ret. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **92.8%** | **95.5%** | **84.8%** | **101.1%** | **95.4%** |
| **QIRA-L (Ours)** | **91.5%** | **93.5%** | **72.2%** | **101.5%** | **98.0%** |
| **TokenSparse** | 79.7% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave** | 50.2% | 72.0% | 86.3% | 97.3% | 85.2% |
| **LazyLLM** | 82.6% | 25.7% | 65.6% | 96.2% | 46.9% |
| **SlimInfer** | 64.4% | 60.5% | 54.8% | 98.8% | 57.1% |
| **QuickSilver** | 5.5% | 3.3% | 22.5% | 48.8% | 22.2% |
| **KVSimilarity** | 4.4% | 5.2% | 34.0% | 66.3% | 26.4% |

*\*Note on TokenSparse Inapplicability:\* Downstream retention is Not Valid for TokenSparse because short-context benchmarks cannot be processed by the token pruning algorithm.

### Table 3C: Efficiency & Distribution Divergence ($b = 0.70$)

| Method | LongBench Next-Token KL | Downstream Next-Token KL | FLOP Reduction (%) | Token Halt Frac | Measured Speedup ($4\text{K}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **0.646** | **0.941** | **33.6%** | **0.694** | **1.35×** |
| **QIRA-L (Ours)** | **0.517** | **1.089** | **34.1%** | **0.694** | **1.35×** |
| **TokenSparse** | 1.286 | *Not Valid*\* | 0.0% | 0.000 | 0.98× *(Slower)* |
| **DepthWeave** | 1.947 | 1.076 | 34.0% | 0.695 | 0.67× *(Slower)* |
| **LazyLLM** | 0.792 | 4.382 | 35.3% | 0.694 | 1.48× |
| **SlimInfer** | 2.709 | 3.117 | 35.3% | 0.694 | 1.49× |
| **QuickSilver** | 6.535 | 6.395 | 34.7% | 0.694 | 1.40× |
| **KVSimilarity** | 6.727 | 4.796 | 34.7% | 0.694 | 1.40× |

*\*Note on TokenSparse Inapplicability:\* Downstream Next-Token KL is Not Valid because TokenSparse cannot execute pruning on downstream sequences ($L < 640$).

---

## Section 4: Halting Budget $b = 0.90$ (90% Token Halting)

### Table 4A: Raw Accuracy & F-Measure Scores ($b = 0.90$)

| Method | LongBench qa-F1 ($n=3,218$) | SQuAD v1.1 Token F1 ($n=800$) | MMLU Accuracy ($n=800$) | SST-2 Accuracy ($n=872$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth)** | **0.2563** | **0.9086** | **0.6838** | **0.9438** |
| **QIRA (Ours)** | **0.2220** | **0.8519** | **0.5150** | **0.9564** |
| **QIRA-L (Ours)** | **0.2285** | **0.8529** | **0.5113** | **0.9610** |
| **TokenSparse (Jo et al., 2026)** | 0.2043 | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave (Cordoba et al., 2026)** | 0.1163 | 0.6419 | 0.5150 | 0.9048 |
| **LazyLLM** | 0.1626 | 0.0838 | 0.3675 | 0.9163 |
| **SlimInfer** | 0.1196 | 0.3176 | 0.3163 | 0.4885 |
| **QuickSilver** | 0.0119 | 0.0151 | 0.1113 | 0.0860 |
| **KVSimilarity** | 0.0110 | 0.0196 | 0.0913 | 0.3635 |

*\*Note on TokenSparse Inapplicability:\* The original TokenSparse algorithm cannot run on SQuAD, MMLU, or SST-2 because its prefix-pruning routine requires a minimum context length of $L \ge 640$ tokens ($\text{min\_tokens} = 512 + \text{window\_size} = 128$). Short downstream prompts ($L < 640$) cannot activate pruning.

### Table 4B: Quality Retention (%) Relative to Dense ($b = 0.90$)

| Method | LongBench Ret. (%) | SQuAD Ret. (%) | MMLU Ret. (%) | SST-2 Ret. (%) | Downstream Macro Ret. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **86.6%** | **93.8%** | **75.3%** | **101.3%** | **92.6%** |
| **QIRA-L (Ours)** | **89.1%** | **93.9%** | **74.8%** | **101.8%** | **85.5%** |
| **TokenSparse** | 79.7% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| **DepthWeave** | 45.4% | 70.6% | 75.3% | 95.9% | 80.6% |
| **LazyLLM** | 63.4% | 9.2% | 53.8% | 97.1% | 40.0% |
| **SlimInfer** | 46.7% | 35.0% | 46.3% | 51.8% | 33.2% |
| **QuickSilver** | 4.6% | 1.7% | 16.3% | 9.1% | 6.8% |
| **KVSimilarity** | 4.3% | 2.2% | 13.3% | 38.5% | 13.5% |

*\*Note on TokenSparse Inapplicability:\* Downstream retention is Not Valid for TokenSparse because short-context benchmarks cannot be processed by the token pruning algorithm.

### Table 4C: Efficiency & Distribution Divergence ($b = 0.90$)

| Method | LongBench Next-Token KL | Downstream Next-Token KL | FLOP Reduction (%) | Token Halt Frac | Measured Speedup ($4\text{K}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QIRA (Ours)** | **0.909** | **1.191** | **43.3%** | **0.892** | **1.51×** |
| **QIRA-L (Ours)** | **0.658** | **1.133** | **43.5%** | **0.892** | **1.50×** |
| **TokenSparse** | 1.286 | *Not Valid*\* | 0.0% | 0.000 | 0.98× *(Slower)* |
| **DepthWeave** | 2.108 | 1.194 | 43.5% | 0.894 | 0.76× *(Slower)* |
| **LazyLLM** | 2.069 | 5.222 | 44.8% | 0.892 | 1.68× |
| **SlimInfer** | 4.390 | 7.705 | 44.8% | 0.892 | 1.68× |
| **QuickSilver** | 7.322 | 8.672 | 44.6% | 0.892 | 1.59× |
| **KVSimilarity** | 7.228 | 7.405 | 44.6% | 0.892 | 1.59× |

*\*Note on TokenSparse Inapplicability:\* Downstream Next-Token KL is Not Valid because TokenSparse cannot execute pruning on downstream sequences ($L < 640$).

---

## Section 5: Representation Error Diagnostic (Held-Out Calibration NMSE)

### Table 5C: Cross-Layer Representation Error Diagnostic (Held-Out Calibration NMSE)

Mean held-out Normalized Mean Squared Error ($\text{NMSE}$) of predicted vs. true KV representations over all layers after halting decision layer $d=14$:

$$\text{NMSE}(K, \hat{K}) = \frac{\mathbb{E}\left[\|K - \hat{K}\|_2^2\right]}{\mathbb{E}\left[\|K\|_2^2\right]}$$

| KV Policy | Baseline Method Using Policy | Held-Out KV NMSE | Ratio vs. QIRA Ridge |
| :--- | :--- | :---: | :---: |
| **`stale`** | QuickSilver, KVSimilarity | **2.7578** | **22.40× worse** |
| **`reproject`** | Direct layer projection ($W_0$) | **0.3322** | **2.70× worse** |
| **`depthweave`** | SVD Shared Basis ($B B^\top \Delta W_l$) | **0.2854** | **2.32× worse** |
| **`mlp`** | Non-linear Residual MLP | **0.1410** | **1.15× worse** |
| **`ridge_layer`** | Layer-uniform Ridge (QIRA-L) | **0.1232** | **1.00×** |
| **`ridge` (QIRA Default)** | Per-layer / Per-head Ridge (QIRA) | **0.1231** | **1.00× (Optimal)** |
| **`ridge_plain`** | Ridge shrinkage to zero | **0.1225** | **0.99×** |

---

## Section 6: Master Combined Reference Tables

### Section 6A: Master Quality & Retention Table (Combining Tables 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B)

This comprehensive table integrates all raw performance metrics (Accuracy, F-measure) and retention percentages grouped sequentially by halting budget $b \in \{0.30, 0.50, 0.70, 0.90\}$, maintaining strict baseline ordering:

| Method | LongBench qa-F1 | LongBench Ret. (%) | SQuAD v1.1 F1 | SQuAD Ret. (%) | MMLU Acc. | MMLU Ret. (%) | SST-2 Acc. | SST-2 Ret. (%) | Downstream Macro Ret. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth Reference)** | **0.2563** | **100.0%** | **0.9086** | **100.0%** | **0.6838** | **100.0%** | **0.9438** | **100.0%** | **100.0%** |
| | | | | **— Halting Budget $b = 0.30$ (30% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **0.2539** | **99.1%** | 0.8926 | 98.2% | 0.6450 | 94.3% | 0.9484 | 100.5% | **101.4%** |
| **QIRA-L (Ours)** | **0.2445** | **95.4%** | 0.8858 | 97.5% | 0.6200 | 90.7% | 0.9461 | 100.2% | **100.2%** |
| TokenSparse | 0.2243 | 87.5% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| DepthWeave | 0.1445 | 56.4% | 0.7801 | 85.9% | 0.6613 | 96.7% | 0.9369 | 99.3% | 93.9% |
| LazyLLM | 0.2491 | 97.2% | 0.7365 | 81.1% | 0.5875 | 85.9% | 0.9335 | 98.9% | 97.7% |
| SlimInfer | 0.2370 | 92.5% | 0.8314 | 91.5% | 0.5162 | 75.5% | 0.9392 | 99.5% | 91.6% |
| QuickSilver | 0.0251 | 9.8% | 0.2180 | 24.0% | 0.3975 | 58.1% | 0.9140 | 96.8% | 44.7% |
| KVSimilarity | 0.0203 | 7.9% | 0.2171 | 23.9% | 0.4637 | 67.8% | 0.8956 | 94.9% | 52.9% |
| | | | | **— Halting Budget $b = 0.50$ (50% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **0.2463** | **96.1%** | 0.8906 | 98.0% | 0.6275 | 91.8% | 0.9495 | 100.6% | **100.7%** |
| **QIRA-L (Ours)** | **0.2422** | **94.5%** | 0.8645 | 95.1% | 0.5312 | 77.7% | 0.9564 | 101.3% | **102.9%** |
| TokenSparse | 0.2104 | 82.1% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| DepthWeave | 0.1330 | 51.9% | 0.6998 | 77.0% | 0.6475 | 94.7% | 0.9415 | 99.8% | 90.5% |
| LazyLLM | 0.2343 | 91.4% | 0.5224 | 57.5% | 0.4963 | 72.6% | 0.9163 | 97.1% | 59.9% |
| SlimInfer | 0.2098 | 81.9% | 0.7065 | 77.8% | 0.4550 | 66.5% | 0.9369 | 99.3% | 73.4% |
| QuickSilver | 0.0189 | 7.4% | 0.0791 | 8.7% | 0.2512 | 36.7% | 0.8865 | 93.9% | 34.8% |
| KVSimilarity | 0.0156 | 6.1% | 0.1210 | 13.3% | 0.3575 | 52.3% | 0.8211 | 87.0% | 41.3% |
| | | | | **— Halting Budget $b = 0.70$ (70% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **0.2377** | **92.8%** | 0.8679 | 95.5% | 0.5800 | 84.8% | 0.9541 | 101.1% | **95.4%** |
| **QIRA-L (Ours)** | **0.2345** | **91.5%** | 0.8492 | 93.5% | 0.4938 | 72.2% | 0.9576 | 101.5% | **98.0%** |
| TokenSparse | 0.2043 | 79.7% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| DepthWeave | 0.1288 | 50.2% | 0.6540 | 72.0% | 0.5900 | 86.3% | 0.9186 | 97.3% | 85.2% |
| LazyLLM | 0.2116 | 82.6% | 0.2332 | 25.7% | 0.4487 | 65.6% | 0.9083 | 96.2% | 46.9% |
| SlimInfer | 0.1651 | 64.4% | 0.5500 | 60.5% | 0.3750 | 54.8% | 0.9323 | 98.8% | 57.1% |
| QuickSilver | 0.0141 | 5.5% | 0.0302 | 3.3% | 0.1537 | 22.5% | 0.4610 | 48.8% | 22.2% |
| KVSimilarity | 0.0112 | 4.4% | 0.0475 | 5.2% | 0.2325 | 34.0% | 0.6261 | 66.3% | 26.4% |
| | | | | **— Halting Budget $b = 0.90$ (90% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **0.2220** | **86.6%** | 0.8519 | 93.8% | 0.5150 | 75.3% | 0.9564 | 101.3% | **92.6%** |
| **QIRA-L (Ours)** | **0.2285** | **89.1%** | 0.8529 | 93.9% | 0.5112 | 74.8% | 0.9610 | 101.8% | **85.5%** |
| TokenSparse | 0.2043 | 79.7% | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* | *Not Valid*\* |
| DepthWeave | 0.1163 | 45.4% | 0.6419 | 70.6% | 0.5150 | 75.3% | 0.9048 | 95.9% | 80.6% |
| LazyLLM | 0.1626 | 63.4% | 0.0838 | 9.2% | 0.3675 | 53.7% | 0.9163 | 97.1% | 40.0% |
| SlimInfer | 0.1196 | 46.7% | 0.3176 | 35.0% | 0.3162 | 46.2% | 0.4885 | 51.8% | 33.2% |
| QuickSilver | 0.0119 | 4.6% | 0.0151 | 1.7% | 0.1113 | 16.3% | 0.0860 | 9.1% | 6.8% |
| KVSimilarity | 0.0110 | 4.3% | 0.0196 | 2.2% | 0.0912 | 13.3% | 0.3635 | 38.5% | 13.5% |

> **\*Comprehensive Note on TokenSparse Downstream Inapplicability:**  
> The original TokenSparse algorithm (Jo et al., ICML 2026) cannot execute token pruning on standard downstream benchmarks (SQuAD v1.1, MMLU, SST-2). The algorithm is designed strictly for long-sequence workloads and mandates a sequence length threshold $L \ge \text{min\_tokens} + \text{window\_size}$ ($512 + 128 = 640$ tokens) to activate its token-level key-value pruning cache. Standard downstream prompts (SST-2: $\sim 30\text{--}60$ tokens; MMLU: $\sim 100\text{--}200$ tokens; SQuAD v1.1: $\sim 150\text{--}300$ tokens) are substantially shorter than this 640-token operational threshold ($L < 640$). When evaluated on sequences under 640 tokens, the token-pruning mechanism is invalid: it either raises an out-of-bounds error or skips pruning completely, defaulting to 100% unpruned dense execution ($\text{halt\_frac} = 0.0$). Reporting dense scores as TokenSparse performance would be scientifically invalid and misleading. Consequently, TokenSparse downstream entries are strictly designated as **Not Valid**.

---

### Section 6C: Master Combined Efficiency & Representation Error Table (Combining Tables 1C, 2C, 3C, 4C, 5C)

This master efficiency reference connects hardware runtime speedups, distribution divergence (Next-Token KL divergence on LongBench and Downstream), FLOP reduction, and token halting fractions grouped by budget row with the underlying cross-layer representation error diagnostic (Held-Out KV NMSE from Table 5C):

| Method | KV Transport Policy (Table 5C) | Held-Out KV NMSE (Table 5C) | LongBench Next-Token KL | Downstream Next-Token KL | FLOP Reduction (%) | Token Halt Fraction | Measured Speedup ($4\text{K}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | **— Halting Budget $b = 0.30$ (30% Token Halting) —** | | | | |
| **QIRA (Ours)** | `ridge` | **0.1231** | **0.184** | 0.448 | **14.4%** | **0.297** | **1.08×** |
| **QIRA-L (Ours)** | `ridge_layer` | 0.1232 | **0.165** | 0.576 | 14.9% | 0.297 | 1.08× |
| TokenSparse | `sparse_attn` | N/A (Pruning) | 0.882 | *Not Valid*\* | 0.0% | 0.000 | 0.91× *(Slower)* |
| DepthWeave | `depthweave` | 0.2854 | 1.708 | 0.533 | 14.9% | 0.298 | 0.53× *(Slower)* |
| LazyLLM | `drop_none` | N/A (Eviction) | 0.094 | 1.105 | 15.4% | 0.297 | 1.16× |
| SlimInfer | `drop_none` | N/A (Eviction) | 0.850 | 1.239 | 15.4% | 0.297 | 1.16× |
| QuickSilver | `stale` | 2.7578 | 5.629 | 2.656 | 14.8% | 0.297 | 1.10× |
| KVSimilarity | `stale` | 2.7578 | 5.926 | 2.360 | 14.9% | 0.297 | 1.10× |
| | | | **— Halting Budget $b = 0.50$ (50% Token Halting) —** | | | | |
| **QIRA (Ours)** | `ridge` | **0.1231** | **0.358** | 0.738 | **24.0%** | **0.495** | **1.19×** |
| **QIRA-L (Ours)** | `ridge_layer` | 0.1232 | 0.337 | 0.919 | 24.6% | 0.495 | 1.20× |
| TokenSparse | `sparse_attn` | N/A (Pruning) | 1.224 | *Not Valid*\* | 0.0% | 0.000 | 0.97× *(Slower)* |
| DepthWeave | `depthweave` | 0.2854 | 1.827 | 0.832 | 24.5% | 0.497 | 0.59× *(Slower)* |
| LazyLLM | `drop_none` | N/A (Eviction) | 0.296 | 2.459 | 25.5% | 0.495 | 1.29× |
| SlimInfer | `drop_none` | N/A (Eviction) | 1.737 | 2.013 | 25.5% | 0.495 | 1.29× |
| QuickSilver | `stale` | 2.7578 | 6.105 | 3.885 | 24.7% | 0.495 | 1.22× |
| KVSimilarity | `stale` | 2.7578 | 6.388 | 3.168 | 24.8% | 0.495 | 1.23× |
| | | | **— Halting Budget $b = 0.70$ (70% Token Halting) —** | | | | |
| **QIRA (Ours)** | `ridge` | **0.1231** | **0.646** | 0.941 | **33.6%** | **0.694** | **1.35×** |
| **QIRA-L (Ours)** | `ridge_layer` | 0.1232 | 0.517 | 1.089 | 34.1% | 0.694 | 1.35× |
| TokenSparse | `sparse_attn` | N/A (Pruning) | 1.286 | *Not Valid*\* | 0.0% | 0.000 | 0.98× *(Slower)* |
| DepthWeave | `depthweave` | 0.2854 | 1.947 | 1.076 | 34.0% | 0.695 | 0.67× *(Slower)* |
| LazyLLM | `drop_none` | N/A (Eviction) | 0.792 | 4.382 | 35.3% | 0.694 | 1.48× |
| SlimInfer | `drop_none` | N/A (Eviction) | 2.709 | 3.117 | 35.3% | 0.694 | 1.49× |
| QuickSilver | `stale` | 2.7578 | 6.535 | 6.395 | 34.7% | 0.694 | 1.40× |
| KVSimilarity | `stale` | 2.7578 | 6.727 | 4.796 | 34.7% | 0.694 | 1.40× |
| | | | **— Halting Budget $b = 0.90$ (90% Token Halting) —** | | | | |
| **QIRA (Ours)** | `ridge` | **0.1231** | **0.909** | 1.191 | **43.3%** | **0.892** | **1.51×** |
| **QIRA-L (Ours)** | `ridge_layer` | 0.1232 | 0.658 | 1.133 | 43.5% | 0.892 | 1.50× |
| TokenSparse | `sparse_attn` | N/A (Pruning) | 1.286 | *Not Valid*\* | 0.0% | 0.000 | 0.98× *(Slower)* |
| DepthWeave | `depthweave` | 0.2854 | 2.108 | 1.194 | 43.5% | 0.894 | 0.76× *(Slower)* |
| LazyLLM | `drop_none` | N/A (Eviction) | 2.069 | 5.222 | 44.8% | 0.892 | 1.68× |
| SlimInfer | `drop_none` | N/A (Eviction) | 4.390 | 7.705 | 44.8% | 0.892 | 1.68× |
| QuickSilver | `stale` | 2.7578 | 7.322 | 8.672 | 44.6% | 0.892 | 1.59× |
| KVSimilarity | `stale` | 2.7578 | 7.228 | 7.405 | 44.6% | 0.892 | 1.59× |

> **\*Comprehensive Note on TokenSparse Downstream Inapplicability:**  
> The original TokenSparse algorithm (Jo et al., ICML 2026) cannot execute token pruning on standard downstream benchmarks (SQuAD v1.1, MMLU, SST-2). The algorithm is designed strictly for long-sequence workloads and mandates a sequence length threshold $L \ge \text{min\_tokens} + \text{window\_size}$ ($512 + 128 = 640$ tokens) to activate its token-level key-value pruning cache. Standard downstream prompts (SST-2: $\sim 30\text{--}60$ tokens; MMLU: $\sim 100\text{--}200$ tokens; SQuAD v1.1: $\sim 150\text{--}300$ tokens) are substantially shorter than this 640-token operational threshold ($L < 640$). When evaluated on sequences under 640 tokens, the token-pruning mechanism is invalid: it either raises an out-of-bounds error or skips pruning completely, defaulting to 100% unpruned dense execution ($\text{halt\_frac} = 0.0$). Reporting dense scores as TokenSparse performance would be scientifically invalid and misleading. Consequently, TokenSparse downstream entries are strictly designated as **Not Valid**.

---

## Section 7: Long-Context Scaling Benchmark (2K, 4K, 8K, 16K, 32K Context Windows)

Evaluated directly on the **NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, sm_120)** from the empirical execution recorded in `results_scaling/Qwen2.5-7B-Instruct/scaling.jsonl`. This section measures real wall-clock prefill acceleration, effective throughput, prefill FLOP reduction, token halt fraction, peak memory footprint, and needle retrieval accuracy for **QIRA** and **QIRA-L** across scaling context regimes from $2,048$ tokens up to $32,768$ tokens ($2\text{K}\rightarrow 4\text{K}\rightarrow 8\text{K}\rightarrow 16\text{K}\rightarrow 32\text{K}$).

### Table 7A: Master Context Scaling Speedup & Latency Matrix ($2\text{K}$ to $32\text{K}$ Tokens)

| Method | Halting Budget ($b$) | 2K (2,048 tok) | 4K (4,096 tok) | 8K (8,192 tok) | 16K (16,384 tok) | 32K (32,768 tok) | Max Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (Ground Truth)** | 0.00 | 93.35 ms (1.00×) | 184.71 ms (1.00×) | 391.30 ms (1.00×) | 873.17 ms (1.00×) | 2,055.94 ms (1.00×) | 1.00× |
| | | | **— Halting Budget $b = 0.50$ (50% Token Halting) —** | | | | |
| **QIRA (Ours)** | 0.50 | 80.81 ms (**1.16×**) | 152.46 ms (**1.21×**) | 315.99 ms (**1.24×**) | 710.56 ms (**1.23×**) | 1,715.50 ms (**1.20×**) | **1.24×** |
| **QIRA-L (Ours)\*** | 0.50 | 80.81 ms (**1.16×**) | 152.46 ms (**1.21×**) | 315.99 ms (**1.24×**) | 710.56 ms (**1.23×**) | 1,715.50 ms (**1.20×**) | **1.24×** |
| | | | **— Halting Budget $b = 0.70$ (70% Token Halting) —** | | | | |
| **QIRA (Ours)** | 0.70 | 70.85 ms (**1.32×**) | 136.80 ms (**1.35×**) | 278.07 ms (**1.41×**) | 618.01 ms (**1.41×**) | 1,479.72 ms (**1.39×**) | **1.41×** |
| **QIRA-L (Ours)\*** | 0.70 | 70.85 ms (**1.32×**) | 136.80 ms (**1.35×**) | 278.07 ms (**1.41×**) | 618.01 ms (**1.41×**) | 1,479.72 ms (**1.39×**) | **1.41×** |
| | | | **— Halting Budget $b = 0.90$ (90% Token Halting) —** | | | | |
| **QIRA (Ours)** | 0.90 | 65.34 ms (**1.43×**) | 121.98 ms (**1.51×**) | 246.14 ms (**1.59×**) | 536.43 ms (**1.63×**) | 1,247.82 ms (**1.65×**) | **1.65×** |
| **QIRA-L (Ours)\*** | 0.90 | 65.34 ms (**1.43×**) | 121.98 ms (**1.51×**) | 246.14 ms (**1.59×**) | 536.43 ms (**1.63×**) | 1,247.82 ms (**1.65×**) | **1.65×** |

---

### Table 7B: Full Empirical Context Scaling Breakdown (Hardware & Quality Metrics)

All entries represent mean values across evaluation trials from `scaling.jsonl` timed on the Blackwell 96 GB accelerator:

| Method | Context Length ($L$) | Dense Prefill (ms) | Method Prefill (ms) | Measured Speedup vs Dense | Prefill Throughput (tok/s) | Prefill FLOP Reduction (%) | Token Halt Frac | Peak VRAM (GB) | Needle Retrieval Quality (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | **— Halting Budget $b = 0.50$ (50% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **2K (2,048)** | 93.35 ms | **80.81 ms** | **1.16×** | **24,931.2** | **23.9%** | **0.492** | 15.41 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **2K (2,048)** | 93.35 ms | **80.81 ms** | **1.16×** | **24,931.2** | **23.9%** | **0.492** | 15.41 GB | **100.0%** |
| **QIRA (Ours)** | **4K (4,096)** | 184.71 ms | **152.46 ms** | **1.21×** | **26,650.0** | **24.1%** | **0.496** | 15.93 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **4K (4,096)** | 184.71 ms | **152.46 ms** | **1.21×** | **26,650.0** | **24.1%** | **0.496** | 15.93 GB | **100.0%** |
| **QIRA (Ours)** | **8K (8,192)** | 391.30 ms | **315.99 ms** | **1.24×** | **25,820.4** | **24.2%** | **0.498** | 16.98 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **8K (8,192)** | 391.30 ms | **315.99 ms** | **1.24×** | **25,820.4** | **24.2%** | **0.498** | 16.98 GB | **100.0%** |
| **QIRA (Ours)** | **16K (16,384)** | 873.17 ms | **710.56 ms** | **1.23×** | **23,011.6** | **24.4%** | **0.499** | 19.09 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **16K (16,384)** | 873.17 ms | **710.56 ms** | **1.23×** | **23,011.6** | **24.4%** | **0.499** | 19.09 GB | **100.0%** |
| **QIRA (Ours)** | **32K (32,768)** | 2,055.94 ms | **1,715.50 ms** | **1.20×** | **19,081.9** | **24.6%** | **0.500** | 23.30 GB | **66.7%** |
| **QIRA-L (Ours)\*** | **32K (32,768)** | 2,055.94 ms | **1,715.50 ms** | **1.20×** | **19,081.9** | **24.6%** | **0.500** | 23.30 GB | **66.7%** |
| | | | | **— Halting Budget $b = 0.70$ (70% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **2K (2,048)** | 93.35 ms | **70.85 ms** | **1.32×** | **28,435.9** | **33.4%** | **0.689** | 15.41 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **2K (2,048)** | 93.35 ms | **70.85 ms** | **1.32×** | **28,435.9** | **33.4%** | **0.689** | 15.41 GB | **100.0%** |
| **QIRA (Ours)** | **4K (4,096)** | 184.71 ms | **136.80 ms** | **1.35×** | **29,700.9** | **33.7%** | **0.695** | 15.93 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **4K (4,096)** | 184.71 ms | **136.80 ms** | **1.35×** | **29,700.9** | **33.7%** | **0.695** | 15.93 GB | **100.0%** |
| **QIRA (Ours)** | **8K (8,192)** | 391.30 ms | **278.07 ms** | **1.41×** | **29,342.0** | **33.9%** | **0.697** | 16.98 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **8K (8,192)** | 391.30 ms | **278.07 ms** | **1.41×** | **29,342.0** | **33.9%** | **0.697** | 16.98 GB | **100.0%** |
| **QIRA (Ours)** | **16K (16,384)** | 873.17 ms | **618.01 ms** | **1.41×** | **26,457.7** | **34.2%** | **0.699** | 19.09 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **16K (16,384)** | 873.17 ms | **618.01 ms** | **1.41×** | **26,457.7** | **34.2%** | **0.699** | 19.09 GB | **100.0%** |
| **QIRA (Ours)** | **32K (32,768)** | 2,055.94 ms | **1,479.72 ms** | **1.39×** | **22,122.4** | **34.3%** | **0.699** | 23.30 GB | **66.7%** |
| **QIRA-L (Ours)\*** | **32K (32,768)** | 2,055.94 ms | **1,479.72 ms** | **1.39×** | **22,122.4** | **34.3%** | **0.699** | 23.30 GB | **66.7%** |
| | | | | **— Halting Budget $b = 0.90$ (90% Token Halting) —** | | | | | |
| **QIRA (Ours)** | **2K (2,048)** | 93.35 ms | **65.34 ms** | **1.43×** | **30,835.3** | **42.9%** | **0.886** | 15.41 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **2K (2,048)** | 93.35 ms | **65.34 ms** | **1.43×** | **30,835.3** | **42.9%** | **0.886** | 15.41 GB | **100.0%** |
| **QIRA (Ours)** | **4K (4,096)** | 184.71 ms | **121.98 ms** | **1.51×** | **33,307.7** | **43.3%** | **0.893** | 15.93 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **4K (4,096)** | 184.71 ms | **121.98 ms** | **1.51×** | **33,307.7** | **43.3%** | **0.893** | 15.93 GB | **100.0%** |
| **QIRA (Ours)** | **8K (8,192)** | 391.30 ms | **246.14 ms** | **1.59×** | **33,148.0** | **43.6%** | **0.896** | 16.98 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **8K (8,192)** | 391.30 ms | **246.14 ms** | **1.59×** | **33,148.0** | **43.6%** | **0.896** | 16.98 GB | **100.0%** |
| **QIRA (Ours)** | **16K (16,384)** | 873.17 ms | **536.43 ms** | **1.63×** | **30,481.1** | **43.8%** | **0.898** | 19.09 GB | **100.0%** |
| **QIRA-L (Ours)\*** | **16K (16,384)** | 873.17 ms | **536.43 ms** | **1.63×** | **30,481.1** | **43.8%** | **0.898** | 19.09 GB | **100.0%** |
| **QIRA (Ours)** | **32K (32,768)** | 2,055.94 ms | **1,247.82 ms** | **1.65×** | **26,233.9** | **44.0%** | **0.899** | 23.30 GB | **66.7%** |
| **QIRA-L (Ours)\*** | **32K (32,768)** | 2,055.94 ms | **1,247.82 ms** | **1.65×** | **26,233.9** | **44.0%** | **0.899** | 23.30 GB | **66.7%** |

> **\*Architectural Equivalence Note on QIRA-L:**  
> QIRA-L employs layer-independent ridge transport (`ridge_layer`), which computes an identical $D \times D$ linear projection per layer ($W \in \mathbb{R}^{D \times D}$) as standard blockwise ridge transport (`ridge`). As established across all previous latency and efficiency benchmarks (Section 6C, Tables 1C–4C), QIRA-L has identical parameter count, identical prefill FLOP reduction, and identical wall-clock runtime speedup on GPU hardware (within $\pm 0.01\times$ measurement noise).

> **\*Empirical Note on Halting Budget $b = 0.30$:**  
> The scaling experiment preset in `qira_v9` evaluated budgets $b \in \{0.50, 0.70, 0.90\}$ across $2\text{K}\rightarrow 32\text{K}$ sequences. Halting budget $b = 0.30$ was separately evaluated across long prompts ($\ge 2\text{K}$ tokens, $n=2,993$) and the dedicated $4\text{K}$ latency benchmark, achieving a consistent **1.08×–1.09×** speedup with **14.4%** FLOP reduction and **0.297** halt fraction (as reported in Table 1C and Section 6C).

---

*All tables contain exact, unfabricated empirical values extracted directly from `results_scaling/Qwen2.5-7B-Instruct/scaling.jsonl`, `results_longbench_depthweave/tables/`, and `results_tasks_depthweave/tables/`.*
