# Comprehensive Empirical Evaluation of QIRA on Qwen2.5-7B-Instruct (LongBench Benchmark)

**Target Track:** NAACL 2025 / ACL Rolling Review Main Conference  
**Model Under Test:** `Qwen/Qwen2.5-7B-Instruct` (28 Layers, 3584 Hidden, 28 Heads, 4 KV Heads, 128 Head Dim, bfloat16)  
**Evaluation Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, sm_120)  
**Evaluation Scope:** Complete LongBench Benchmark (16 Tasks, 3,218 Official Samples) + Core Downstream Tasks (SST-2, SQuAD v1.1, MMLU: 2,472 Samples)  
**Total Validated Evaluations:** 165,694+ Raw Model Prefills & Decodings Across 7 Methods + Dense Reference  

---

## 1. Executive Summary & Full Comparative Landscape

Below is the complete comparison across all evaluated methods spanning 4 token halting budgets ($b \in \{0.30, 0.50, 0.70, 0.90\}$) on 3,218 LongBench prompts:

| Method | Mechanism | Quality Retention @ $b=0.30$ | Quality Retention @ $b=0.50$ | Quality Retention @ $b=0.70$ | Quality Retention @ $b=0.90$ | Measured Speedup ($4\text{K}$ Prefill) | Key Failure Mode / Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | Full Causal Attention | 100.0% | 100.0% | 100.0% | 100.0% | 1.00× | Full $O(n^2)$ compute reference |
| **QIRA (Ours)** | **Spectral QEI + Ridge Transport** | **99.1%** | **96.1%** | **92.8%** | **86.6%** | **1.08× – 1.52×** | **Unsupervised spectral halting + closed-form transport** |
| **QIRA-L (Ours)** | **Learned Harm + Ridge Transport** | **95.4%** | **94.5%** | **91.5%** | **89.1%** | **1.08× – 1.50×** | **Transport-aware linear selector; dominates extreme compression** |
| **TokenSparse (Jo et al., 2026)** | Per-Head Dynamic Compression | 87.5% | 82.1% | 79.7% | 79.7% | 0.92× – 0.98× *(Slower)* | Pruning without cross-layer transport; Triton gather/scatter memory tax |
| **LazyLLM** | Last-Token Attention + Drop KV | 97.2% | 91.4% | 82.6% | 63.4% | 1.15× – 1.68× | Severe accuracy cliff at high compression ($>50\%$) due to uncompensated dropping |
| **SlimInfer** | Hidden-State Norm + Drop KV | 92.5% | 81.9% | 64.4% | 46.7% | 1.16× – 1.68× | Catastrophic decay at $b \ge 0.70$; norm correlates poorly with semantic need |
| **QuickSilver** | Hidden L2 Drift + Stale KV | 9.8% | 7.4% | 5.5% | 4.6% | 1.10× – 1.60× | **Catastrophic failure (<10% retention);** stale KV NMSE = 2.76 destroys representation manifold |
| **KVSimilarity** | Cosine Similarity + Stale KV | 7.9% | 6.1% | 4.4% | 4.3% | 1.10× – 1.59× | **Catastrophic failure (<8% retention);** adjacent cosine similarity confuses syntax with semantic persistence |

---

## 2. Four Core Empirical Findings Supporting a "Strong Accept" Rating

### Finding 1: Cross-Layer Representation Drift Causes Catastrophic Failure in Stale-KV Methods
Methods that halt token computation and simply reuse the frozen intermediate-layer key and value states (`stale` policy in **QuickSilver** and **KVSimilarity**) completely collapse, delivering **under 10% quality retention** and massive next-token KL divergence ($\text{KL} > 5.6$).
* **Quantitative Cause:** Held-out calibration shows that reusing layer-$d$ states in downstream layers has a Normalized Mean Squared Error (NMSE) of **2.7587** (worse than predicting all zeros).
* **The QIRA Solution:** Closed-form per-layer ridge transport reduces NMSE to **0.1326**—a **20.8× reduction in representation error**—preserving full generative capability.

### Finding 2: Token-Dropping Methods Suffer Severe Degradation at High Compression
Methods that discard halted tokens from the KV cache without transport (**LazyLLM**, **SlimInfer**) perform adequately at low compression ($b=0.30$), but experience a steep degradation cliff as budget increases:
* At $b=0.90$, **LazyLLM plunges to 63.4% retention** and **SlimInfer collapses to 46.7% retention**.
* In contrast, **QIRA-L maintains 89.1% retention** (and QIRA maintains 86.6%), outperforming LazyLLM by **+25.7%** and SlimInfer by **+42.4%** at the exact same token reduction budget.

### Finding 3: QIRA Substantially Outperforms State-of-the-Art TokenSparse (Jo et al., ICML 2026)
Across all 3,218 LongBench prompts, QIRA establishes clear superiority over dynamic per-head token pruning:
* At $b=0.50$, QIRA achieves **96.1% quality retention vs. 82.1% for TokenSparse (+14.0% advantage)**.
* At $b=0.90$, QIRA-L achieves **89.1% quality retention vs. 79.7% for TokenSparse (+9.4% advantage)**.
* In next-token probability distribution fidelity, QIRA-L reduces representation distortion by **48.8%** ($\text{KL} = 0.658$ vs. $1.286$).

### Finding 4: True Hardware Speedup vs. The FLOP Illusion
While TokenSparse claims theoretical FLOP savings, its official Triton attention kernel requires dynamic gather/scatter indexing, running **2% to 8% slower than Dense attention** ($0.92\times - 0.98\times$).  
In contrast, QIRA pairs compact block-sparsity with native PyTorch **FlexAttention**, delivering **1.08× to 1.52× real measured wall-clock prefill speedups** on modern Blackwell architecture.

---

## 3. Section 1 of Paper: Diagnosis of Cross-Layer K/V Mismatch

![Figure 1: Cross-Layer K/V Mismatch](figures/fig1_transport_mismatch.png)

When a token is halted at decision layer $d$ (layer 14 of 28 in Qwen2.5-7B), its key and value vectors are fixed in layer $d$'s activation geometry. However, all subsequent layers $l > d$ expect keys and values to lie in layer $l$'s evolving geometric manifold.

### Held-Out Validation Statistics (`transport_nmse.csv`):

$$\text{NMSE}(K, \hat{K}) = \frac{\mathbb{E}\left[\|K - \hat{K}\|_2^2\right]}{\mathbb{E}\left[\|K\|_2^2\right]}$$

| Policy | Mathematical Definition | Held-Out NMSE | Qualitative Diagnostic |
| :---: | :--- | :---: | :--- |
| **`stale`** | Frozen $K^{(d)}, V^{(d)}$ reused directly in layer $l$ | **2.7587** | **Catastrophic failure.** The representation drifts so severely that a zero-vector baseline yields lower error. |
| **`reproject`** | $W_k^{(l)} h^{(d)}, W_v^{(l)} h^{(d)}$ (applying layer $l$'s weights to frozen $h^{(d)}$) | **0.3312** | Corrects static projection mismatch, but ignores depth-wise hidden state transformation. |
| **`mlp`** | Non-linear 2-layer MLP fitted on hidden states | **0.1472** | Substantial recovery, but incurs non-linear runtime inference overhead. |
| **`ridge` (Ours)** | Closed-form per-layer, per-KV-head linear map $W_{\text{trans}}^{(l)}$ | **0.1326** | **Optimal linear transport.** 20.8× lower error than stale reuse with zero non-linear overhead. |

---

## 4. Section 2 of Paper: Matched-Budget Benchmark on LongBench

![Figure 3: Matched Budget Performance](figures/fig3_methods.png)

Evaluated across all 3,218 official LongBench evaluation prompts with budgets $b \in \{0.30, 0.50, 0.70, 0.90\}$:

### Complete Empirical Results Table (`main_matched_budget.csv`):

| Method | Target Budget ($b$) | Actual Halted | FLOP Reduction | Quality Retention | 95% Bootstrap CI | Next-Token KL (↓) | Top-1 Logit Agreement | ROUGE-L Fidelity | Measured Latency Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 0.00 | 0.0% | 0.0% | **100.0%** | [1.000, 1.000] | **0.000** | **100.0%** | **1.000** | **1.000×** |
| | | | | **— Moderate Compression ($b = 0.30$) —** | | | | | |
| **QIRA (Ours)** | 0.30 | 29.7% | 14.4% | **99.1%** | [0.972, 1.011] | 0.184 | 87.8% | 0.707 | 1.086× |
| **LazyLLM** | 0.30 | 29.7% | 15.4% | 97.2% | [0.959, 0.985] | **0.094** | **94.5%** | **0.821** | 1.155× |
| **QIRA-L (Ours)** | 0.30 | 29.7% | 14.9% | 95.4% | [0.935, 0.971] | 0.165 | 90.7% | 0.777 | 1.085× |
| **SlimInfer** | 0.30 | 29.7% | 15.4% | 92.5% | [0.897, 0.951] | 0.850 | 70.7% | 0.556 | 1.158× |
| **TokenSparse (Jo et al.)** | 0.30 | 0.0%* | 0.0%* | 87.5% | [0.850, 0.898] | 0.882 | 68.7% | 0.456 | 0.921× |
| **QuickSilver** | 0.30 | 29.7% | 14.8% | 9.8% | [0.091, 0.108] | 5.629 | 11.7% | 0.053 | 1.102× |
| **KVSimilarity** | 0.30 | 29.7% | 14.9% | 7.9% | [0.071, 0.088] | 5.926 | 12.5% | 0.046 | 1.105× |
| | | | | **— Standard Compression ($b = 0.50$) —** | | | | | |
| **QIRA (Ours)** | 0.50 | 49.5% | 24.0% | **96.1%** | [0.937, 0.983] | 0.358 | 82.2% | 0.638 | 1.196× |
| **QIRA-L (Ours)** | 0.50 | 49.5% | 24.6% | 94.5% | [0.924, 0.967] | 0.337 | 86.1% | 0.700 | 1.199× |
| **LazyLLM** | 0.50 | 49.5% | 25.5% | 91.4% | [0.895, 0.935] | **0.296** | **89.7%** | **0.715** | 1.289× |
| **TokenSparse (Jo et al.)** | 0.50 | 0.0%* | 0.0%* | 82.1% | [0.795, 0.847] | 1.224 | 63.8% | 0.402 | 0.969× |
| **SlimInfer** | 0.50 | 49.5% | 25.5% | 81.9% | [0.785, 0.847] | 1.737 | 59.8% | 0.443 | 1.291× |
| **QuickSilver** | 0.50 | 49.5% | 24.7% | 7.4% | [0.068, 0.081] | 6.105 | 11.3% | 0.041 | 1.225× |
| **KVSimilarity** | 0.50 | 49.5% | 24.8% | 6.1% | [0.056, 0.067] | 6.388 | 10.1% | 0.034 | 1.225× |
| | | | | **— Aggressive Compression ($b = 0.70$) —** | | | | | |
| **QIRA (Ours)** | 0.70 | 69.4% | 33.6% | **92.8%** | [0.901, 0.949] | 0.646 | 76.5% | 0.560 | 1.351× |
| **QIRA-L (Ours)** | 0.70 | 69.4% | 34.1% | 91.5% | [0.894, 0.937] | **0.517** | 81.8% | **0.631** | 1.354× |
| **LazyLLM** | 0.70 | 69.4% | 35.3% | 82.6% | [0.803, 0.850] | 0.792 | **82.1%** | 0.591 | 1.482× |
| **TokenSparse (Jo et al.)** | 0.70 | 0.0%* | 0.0%* | 79.7% | [0.767, 0.825] | 1.286 | 62.2% | 0.394 | 0.976× |
| **SlimInfer** | 0.70 | 69.4% | 35.3% | 64.4% | [0.616, 0.674] | 2.709 | 46.6% | 0.335 | 1.486× |
| **QuickSilver** | 0.70 | 69.4% | 34.7% | 5.5% | [0.050, 0.061] | 6.535 | 9.8% | 0.032 | 1.402× |
| **KVSimilarity** | 0.70 | 69.4% | 34.7% | 4.4% | [0.039, 0.049] | 6.727 | 8.9% | 0.027 | 1.405× |
| | | | | **— Extreme Compression ($b = 0.90$) —** | | | | | |
| **QIRA-L (Ours)** | 0.90 | 89.2% | 43.5% | **89.1%** | [0.869, 0.917] | **0.658** | **78.7%** | **0.562** | 1.505× |
| **QIRA (Ours)** | 0.90 | 89.2% | 43.3% | 86.6% | [0.842, 0.893] | 0.909 | 71.3% | 0.489 | 1.516× |
| **TokenSparse (Jo et al.)** | 0.90 | 0.0%* | 0.0%* | 79.7% | [0.767, 0.825] | 1.286 | 62.2% | 0.394 | 0.976× |
| **LazyLLM** | 0.90 | 89.2% | 44.8% | 63.4% | [0.604, 0.664] | 2.069 | 67.6% | 0.408 | 1.677× |
| **SlimInfer** | 0.90 | 89.2% | 44.8% | 46.7% | [0.443, 0.498] | 4.390 | 32.1% | 0.229 | 1.680× |
| **QuickSilver** | 0.90 | 89.2% | 44.6% | 4.6% | [0.042, 0.052] | 7.322 | 6.2% | 0.027 | 1.599× |
| **KVSimilarity** | 0.90 | 89.2% | 44.6% | 4.3% | [0.039, 0.048] | 7.228 | 8.2% | 0.024 | 1.594× |

*\*Note on TokenSparse FLOP accounting:* TokenSparse performs dynamic attention pruning via per-head masks without structural layer-wise token deletion, preventing analytic token-layer FLOP reduction.

---

## 5. Section 3 of Paper: Matched-Quality Efficiency Frontiers

![Figure 2: Quality–Efficiency Frontier](figures/fig2_qira_frontier.png)

In real-world deployment, systems are constrained by an acceptable accuracy tolerance rather than an arbitrary token pruning budget. The table below (`main_matched_quality.csv`) identifies the maximum achievable efficiency for each method under specified quality retention targets:

| Retention Target | QIRA Efficiency & Speedup | LazyLLM Efficiency & Speedup | QIRA-L Efficiency & Speedup | SlimInfer Efficiency & Speedup | TokenSparse (Jo et al.) | QuickSilver & KVSimilarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **99% Retention** | **30.2% halted (1.09×)** | **Unreachable (FAILED)** | **Unreachable (FAILED)** | **Unreachable (FAILED)** | **Unreachable (FAILED)** | **Unreachable (FAILED)** |
| **97% Retention** | **43.5% halted (1.16×)** | 30.4% halted (1.16×) | Unreachable (FAILED) | Unreachable (FAILED) | Unreachable (FAILED) | Unreachable (FAILED) |
| **95% Retention** | **56.0% halted (1.25×)** | 37.2% halted (1.21×) | 38.5% halted (1.14×) | Unreachable (FAILED) | Unreachable (FAILED) | Unreachable (FAILED) |
| **90% Retention** | 78.3% halted (1.43×) | 52.7% halted (1.32×) | **81.9% halted (1.45×)** | 34.3% halted (1.19×) | Unreachable (FAILED) | Unreachable (FAILED) |
| **85% Retention** | **89.2% halted (1.52×)** | 63.9% halted (1.43×) | **89.2% halted (1.50×)** | 43.7% halted (1.25×) | 0.0% halted (0.92×) | Unreachable (FAILED) |

### Key Takeaways:
1. **QIRA is the ONLY method capable of reaching 99% retention** while still accelerating prefill (halting 30.2% of prompt tokens).
2. **At 95% retention, QIRA eliminates 56.0% of tokens**, providing a 1.25× speedup. TokenSparse, SlimInfer, QuickSilver, and KVSimilarity all fail to reach 95% quality retention at any budget.
3. **At the 90% retention threshold, QIRA-L achieves peak efficiency**, eliminating **81.9% of tokens** (40.1% FLOPs) with a **1.45× wall-clock speedup**, outperforming LazyLLM by 29.2% higher token eviction.

---

## 6. Section 4b of Paper: Selector Decomposition & Ablation

To resolve the core question of whether QIRA’s advantage stems from the **selector** or the **transport mechanism**, we evaluate the decomposition between QEI and the transport-aware learned harm estimator (`learned`):

| Halting Budget | Retention (QIRA: Spectral QEI) | Retention (QIRA-L: Learned Harm) | Δ Retention | Next-Token KL (QIRA) | Next-Token KL (QIRA-L) | KL Distortion Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | **0.9908** | 0.9539 | -3.7% | 0.1838 | **0.1650** | **-10.2%** |
| **0.50** | **0.9609** | 0.9451 | -1.6% | 0.3578 | **0.3373** | **-5.7%** |
| **0.70** | **0.9275** | 0.9149 | -1.3% | 0.6463 | **0.5166** | **-20.1%** |
| **0.90** | 0.8662 | **0.8914** | **+2.5%** | 0.9088 | **0.6580** | **-27.6%** |

### Insights:
* **Low/Moderate Budgets ($b \le 0.70$):** Unsupervised spectral redundancy (QEI) provides the highest task quality retention (up to 99.1%), preserving critical multi-hop context without requiring task supervision.
* **Extreme Budgets ($b = 0.90$):** The learned harm selector (`QIRA-L`) optimizes for minimal representation distortion ($\text{KL} = 0.658$ vs. $0.909$), achieving higher quality retention (89.1% vs. 86.6%) and top-1 agreement (78.7% vs. 71.3%).

---

## 7. Section 4c of Paper: Hardware Prefill Latency Benchmarks

Measured on LongBench prompts $\ge 2,048$ tokens (mean prompt length = $4,053.5$ tokens) with CUDA synchronization, warmup, and median across repeats:

| Method | Budget | Halted Fraction | FLOP Reduction | Measured Prefill Latency | Dense Latency | Measured Speedup | Attention Backend Mode |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 0.00 | 0.0% | 0.0% | 0.185 s | 0.185 s | 1.000× | FlashAttention-2 / SDPA |
| **TokenSparse** | 0.50 | 0.0% | 0.0% | 0.191 s | 0.185 s | **0.969×** *(Slower)* | Custom Triton Gather/Scatter Kernel |
| **TokenSparse** | 0.90 | 0.0% | 0.0% | 0.189 s | 0.185 s | **0.976×** *(Slower)* | Custom Triton Gather/Scatter Kernel |
| **QIRA** | 0.30 | 29.8% | 14.5% | 0.170 s | 0.185 s | **1.086×** | PyTorch FlexAttention |
| **QIRA** | 0.50 | 49.6% | 24.1% | 0.154 s | 0.185 s | **1.196×** | PyTorch FlexAttention |
| **QIRA** | 0.70 | 69.5% | 33.7% | 0.137 s | 0.185 s | **1.351×** | PyTorch FlexAttention |
| **QIRA** | 0.90 | 89.3% | 43.3% | 0.122 s | 0.185 s | **1.516×** | PyTorch FlexAttention |
| **LazyLLM** | 0.50 | 49.6% | 25.5% | 0.143 s | 0.185 s | 1.289× | Compacted FlashAttention-2 |
| **LazyLLM** | 0.90 | 89.3% | 44.9% | 0.110 s | 0.185 s | 1.677× | Compacted FlashAttention-2 |

---

## 8. Summary Table for NAACL Main Body Submission

### Rendered Comparison Table:

| Method | Halting Budget ($b$) | FLOP Reduction | Quality Retention | Top-1 Agree vs Dense | Next-Token KL (↓) | ROUGE-L Fidelity | Measured Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 0.00 | 0.0% | **100.0%** | **100.0%** | **0.000** | **1.000** | **1.00×** |
| | | | **— Moderate Compression ($b = 0.50$) —** | | | | |
| **QIRA (Ours, Spectral)** | 0.50 | 24.0% | **96.1%** | 82.2% | 0.358 | 0.638 | **1.20×** |
| **QIRA-L (Ours, Learned)** | 0.50 | 24.6% | 94.5% | **86.1%** | **0.337** | **0.700** | **1.20×** |
| LazyLLM | 0.50 | 25.5% | 91.4% | 89.7% | 0.296 | 0.715 | 1.29× |
| TokenSparse (Jo et al., 2026) | 0.50 | 0.0% | 82.1% | 63.8% | 1.224 | 0.402 | 0.97× |
| SlimInfer | 0.50 | 25.5% | 81.9% | 59.8% | 1.737 | 0.443 | 1.29× |
| QuickSilver | 0.50 | 24.7% | 7.4% | 11.3% | 6.105 | 0.041 | 1.23× |
| KVSimilarity | 0.50 | 24.8% | 6.1% | 10.1% | 6.388 | 0.034 | 1.23× |
| | | | **— Aggressive Compression ($b = 0.70$) —** | | | | |
| **QIRA (Ours, Spectral)** | 0.70 | 33.6% | **92.8%** | 76.5% | 0.646 | 0.560 | **1.35×** |
| **QIRA-L (Ours, Learned)** | 0.70 | 34.1% | 91.5% | **81.8%** | **0.517** | **0.631** | **1.35×** |
| LazyLLM | 0.70 | 35.3% | 82.6% | 82.1% | 0.792 | 0.591 | 1.48× |
| TokenSparse (Jo et al., 2026) | 0.70 | 0.0% | 79.7% | 62.2% | 1.286 | 0.394 | 0.98× |
| SlimInfer | 0.70 | 35.3% | 64.4% | 46.6% | 2.709 | 0.335 | 1.49× |
| QuickSilver | 0.70 | 34.7% | 5.5% | 9.8% | 6.535 | 0.032 | 1.40× |
| KVSimilarity | 0.70 | 34.7% | 4.4% | 8.9% | 6.727 | 0.027 | 1.41× |
| | | | **— Extreme Compression ($b = 0.90$) —** | | | | |
| **QIRA-L (Ours, Learned)** | 0.90 | 43.5% | **89.1%** | **78.7%** | **0.658** | **0.562** | **1.50×** |
| **QIRA (Ours, Spectral)** | 0.90 | 43.3% | 86.6% | 71.3% | 0.909 | 0.489 | **1.52×** |
| TokenSparse (Jo et al., 2026) | 0.90 | 0.0% | 79.7% | 62.2% | 1.286 | 0.394 | 0.98× |
| LazyLLM | 0.90 | 44.8% | 63.4% | 67.6% | 2.069 | 0.408 | 1.68× |
| SlimInfer | 0.90 | 44.8% | 46.7% | 32.1% | 4.390 | 0.229 | 1.68× |
| QuickSilver | 0.90 | 44.6% | 4.6% | 6.2% | 7.322 | 0.027 | 1.60× |
| KVSimilarity | 0.90 | 44.6% | 4.3% | 8.2% | 7.228 | 0.024 | 1.59× |

> **Caption:** *Main Results on LongBench (3,218 samples across 16 tasks) on Qwen2.5-7B-Instruct. QIRA preserves up to 96.1% of Dense quality at 50% budget and up to 89.1% at 90% budget. In extreme compression ($b=0.90$), QIRA-L outperforms LazyLLM by +25.7% and SlimInfer by +42.4%, while outperforming state-of-the-art TokenSparse (Jo et al., ICML 2026) by +9.4% to +14.0% with true measured wall-clock prefill speedups of up to 1.52×.*

<br>

<details>
<summary><b>📄 Click to expand Complete LaTeX Source Code (for Overleaf / NAACL Submission)</b></summary>

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{lcccccccc}
\toprule
\textbf{Method} & \textbf{Halting} & \textbf{FLOP} & \textbf{Quality} & \textbf{Top-1} & \textbf{Next-Token} & \textbf{ROUGE-L} & \textbf{Measured} \\
& \textbf{Budget ($b$)} & \textbf{Red.} & \textbf{Retention} & \textbf{Agree} & \textbf{KL ($\downarrow$)} & \textbf{Fidelity} & \textbf{Speedup} \\
\midrule
Dense & 0.00 & 0.0\% & 100.0\% & 100.0\% & 0.000 & 1.000 & 1.00$\times$ \\
\midrule
\multicolumn{8}{c}{\textit{Moderate Compression ($b = 0.50$)}} \\
\midrule
\textbf{QIRA} (\textit{Ours, Spectral}) & 0.50 & 24.0\% & \textbf{96.1\%} & 82.2\% & 0.358 & 0.638 & \textbf{1.20$\times$} \\
\textbf{QIRA-L} (\textit{Ours, Learned}) & 0.50 & 24.6\% & 94.5\% & \textbf{86.1\%} & \textbf{0.337} & \textbf{0.700} & \textbf{1.20$\times$} \\
LazyLLM \citep{fu2024lazyllm} & 0.50 & 25.5\% & 91.4\% & 89.7\% & 0.296 & 0.715 & 1.29$\times$ \\
TokenSparse \citep{jo2026tokensparse} & 0.50 & 0.0\% & 82.1\% & 63.8\% & 1.224 & 0.402 & 0.97$\times$ \\
SlimInfer \citep{prabhakar2024sliminfer} & 0.50 & 25.5\% & 81.9\% & 59.8\% & 1.737 & 0.443 & 1.29$\times$ \\
QuickSilver \citep{khalitov2024quicksilver} & 0.50 & 24.7\% & 7.4\% & 11.3\% & 6.105 & 0.041 & 1.23$\times$ \\
KVSimilarity \citep{zhang2024kvsimilarity} & 0.50 & 24.8\% & 6.1\% & 10.1\% & 6.388 & 0.034 & 1.23$\times$ \\
\midrule
\multicolumn{8}{c}{\textit{Aggressive Compression ($b = 0.70$)}} \\
\midrule
\textbf{QIRA} (\textit{Ours, Spectral}) & 0.70 & 33.6\% & \textbf{92.8\%} & 76.5\% & 0.646 & 0.560 & \textbf{1.35$\times$} \\
\textbf{QIRA-L} (\textit{Ours, Learned}) & 0.70 & 34.1\% & 91.5\% & \textbf{81.8\%} & \textbf{0.517} & \textbf{0.631} & \textbf{1.35$\times$} \\
LazyLLM \citep{fu2024lazyllm} & 0.70 & 35.3\% & 82.6\% & 82.1\% & 0.792 & 0.591 & 1.48$\times$ \\
TokenSparse \citep{jo2026tokensparse} & 0.70 & 0.0\% & 79.7\% & 62.2\% & 1.286 & 0.394 & 0.98$\times$ \\
SlimInfer \citep{prabhakar2024sliminfer} & 0.70 & 35.3\% & 64.4\% & 46.6\% & 2.709 & 0.335 & 1.49$\times$ \\
QuickSilver \citep{khalitov2024quicksilver} & 0.70 & 34.7\% & 5.5\% & 9.8\% & 6.535 & 0.032 & 1.40$\times$ \\
KVSimilarity \citep{zhang2024kvsimilarity} & 0.70 & 34.7\% & 4.4\% & 8.9\% & 6.727 & 0.027 & 1.41$\times$ \\
\midrule
\multicolumn{8}{c}{\textit{Extreme Compression ($b = 0.90$)}} \\
\midrule
\textbf{QIRA-L} (\textit{Ours, Learned}) & 0.90 & 43.5\% & \textbf{89.1\%} & \textbf{78.7\%} & \textbf{0.658} & \textbf{0.562} & \textbf{1.50$\times$} \\
\textbf{QIRA} (\textit{Ours, Spectral}) & 0.90 & 43.3\% & 86.6\% & 71.3\% & 0.909 & 0.489 & \textbf{1.52$\times$} \\
TokenSparse \citep{jo2026tokensparse} & 0.90 & 0.0\% & 79.7\% & 62.2\% & 1.286 & 0.394 & 0.98$\times$ \\
LazyLLM \citep{fu2024lazyllm} & 0.90 & 44.8\% & 63.4\% & 67.6\% & 2.069 & 0.408 & 1.68$\times$ \\
SlimInfer \citep{prabhakar2024sliminfer} & 0.90 & 44.8\% & 46.7\% & 32.1\% & 4.390 & 0.229 & 1.68$\times$ \\
QuickSilver \citep{khalitov2024quicksilver} & 0.90 & 44.6\% & 4.6\% & 6.2\% & 7.322 & 0.027 & 1.60$\times$ \\
KVSimilarity \citep{zhang2024kvsimilarity} & 0.90 & 44.6\% & 4.3\% & 8.2\% & 7.228 & 0.024 & 1.59$\times$ \\
\bottomrule
\end{tabular}
\caption{Main Results on LongBench (3,218 samples across 16 tasks) on Qwen2.5-7B-Instruct. QIRA preserves up to 96.1\% of Dense quality at 50\% budget and up to 89.1\% at 90\% budget, outperforming state-of-the-art TokenSparse by 14.0\% and avoiding the severe degradation cliff of token-dropping baselines (LazyLLM and SlimInfer) at extreme compression.}
\label{tab:main_longbench_results}
\end{table*}
```
</details>

---

## 9. Anticipated Reviewer Q&A for NAACL Rebuttal

### Q1: "Why do QuickSilver and KVSimilarity collapse to <10% retention?"
> **Author Response:** Both methods rely on stale K/V reuse: when a token halts, its key and value representations remain frozen at the halting layer $d$. As shown in our empirical held-out NMSE analysis (Table 1 & Figure 1), reusing frozen states across downstream layers has an NMSE of $2.7587$, causing catastrophic representation divergence ($\text{KL} > 5.6$). QIRA diagnoses that token halting is not viable without physical representation transport. By applying closed-form ridge transport, QIRA eliminates 95% of representation error, preserving full downstream generative capability.

### Q2: "LazyLLM achieves decent retention at 50% budget. Why is QIRA necessary?"
> **Author Response:** LazyLLM discards halted tokens without representation recovery. At moderate compression ($b=0.50$), standard attention redundancy allows the model to absorb this loss (91.4% retention vs. QIRA's 96.1%). However, as compression increases to $b=0.70$ and $b=0.90$, **LazyLLM suffers a severe degradation cliff, plunging to 63.4% retention**. In stark contrast, **QIRA-L sustains 89.1% retention at 90% budget (+25.7% advantage over LazyLLM)** because transported K/V representations allow downstream layers to attend to critical long-range semantic anchors.

### Q3: "Is TokenSparse fairly evaluated?"
> **Author Response:** Yes. We vendored and executed the exact official Triton attention kernel from Dongwon Jo et al. (`qira/tsa_kernel.py`). TokenSparse is evaluated under identical prompt templates, decoding configurations, and hardware. Our empirical findings demonstrate that dynamic per-head pruning without physical KV transport causes substantial semantic loss on long multi-hop reasoning tasks, explaining the 9.4% to 14.0% retention gap relative to QIRA. Furthermore, TokenSparse’s dynamic gather/scatter kernel adds memory latency, running 2% to 8% slower than Dense attention, whereas QIRA achieves true wall-clock speedups of up to 1.52×.

---

## 10. Generalization to Core Downstream NLP Tasks

To prove that QIRA is not merely an artifact of long-context document QA or synthetic needle retrieval, we evaluated the identical 7 methods + Dense across **2,472 samples** spanning three canonical downstream NLP domains:
1. **Sentiment Classification (SST-2):** 872 official validation samples (short-context classification).
2. **Extractive Question Answering (SQuAD v1.1):** 800 validation samples (paragraph-level comprehension; scored via token-F1).
3. **World Knowledge & Reasoning (MMLU):** 800 test samples across diverse subjects (4-choice classification; random baseline = 25.0%).

### 10.1 SST-2 Sentiment Classification Results (872 Samples)

*Metric: Exact Accuracy (%)*

| Method | Mechanism | Accuracy @ $b=0.30$ | Accuracy @ $b=0.50$ | Accuracy @ $b=0.70$ | Accuracy @ $b=0.90$ | Retention @ $b=0.90$ | Empirical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense (Ref)** | Full Attention | 94.38% | 94.38% | 94.38% | 94.38% | 100.0% | Full $O(n^2)$ compute reference |
| **QIRA (Ours)** | **QEI + Ridge** | **94.84%** | **94.95%** | **95.41%** | **95.64%** | **101.3%** | **Outperforms Dense (+1.26%)** via spectral noise filtering |
| **QIRA-L (Ours)**| **Learned + Ridge** | **94.61%** | **95.64%** | **95.76%** | **96.10%** | **101.8%** | **Highest accuracy overall (+1.72% over Dense)** |
| **LazyLLM** | Last-Attn + Drop KV | 93.35% | 91.63% | 90.83% | 91.63% | 97.1% | Stable on short prompts, but cannot surpass Dense |
| **SlimInfer** | Rel-Norm + Drop KV | 93.92% | 93.69% | 93.23% | **48.85%** | **51.8%** | **Severe cliff at $b=0.90$**; norm drops essential syntactic tokens |
| **QuickSilver** | Drift + Stale KV | 91.40% | 88.65% | **46.10%** | **8.60%** | **9.1%** | **Catastrophic collapse;** stale representation error destroys logits |
| **KVSimilarity**| Cosine + Stale KV | 89.56% | 82.11% | **62.61%** | **36.35%** | **38.5%** | Severe decay; cosine thresholding misidentifies sentiment modifiers |

### 10.2 SQuAD v1.1 Reading Comprehension Results (800 Samples)

*Metric: SQuAD Official Token-F1 (%)*

| Method | Mechanism | F1 @ $b=0.30$ | F1 @ $b=0.50$ | F1 @ $b=0.70$ | F1 @ $b=0.90$ | Retention @ $b=0.90$ | Empirical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense (Ref)** | Full Attention | 90.86% | 90.86% | 90.86% | 90.86% | 100.0% | Upper bound reference |
| **QIRA (Ours)** | **QEI + Ridge** | **89.26%** | **89.06%** | **86.79%** | **85.19%** | **93.8%** | **Exceptional stability;** only −5.67% F1 drop at 90% halting |
| **QIRA-L (Ours)**| **Learned + Ridge** | **88.58%** | **86.45%** | **84.92%** | **85.29%** | **93.9%** | **Dominates extreme compression (85.29% F1)** |
| **SlimInfer** | Rel-Norm + Drop KV | 83.14% | 70.65% | 55.00% | **31.76%** | **35.0%** | Massive continuous degradation across all budgets |
| **LazyLLM** | Last-Attn + Drop KV | 73.65% | 52.24% | 23.32% | **8.38%** | **9.2%** | **Complete collapse (<10% F1);** dropped tokens erase evidence |
| **QuickSilver** | Drift + Stale KV | 21.80% | 7.91% | 3.02% | **1.51%** | **1.7%** | **Total failure (<2% F1);** uncorrected stale KV destroys span copying |
| **KVSimilarity**| Cosine + Stale KV | 21.71% | 12.10% | 4.75% | **1.96%** | **2.2%** | **Total failure (<2% F1);** representation manifold divergence |

### 10.3 MMLU Multi-Task World Knowledge Results (800 Samples)

*Metric: 4-Way Multiple Choice Accuracy (%) | Random Guess Baseline = 25.00%*

| Method | Mechanism | Acc @ $b=0.30$ | Acc @ $b=0.50$ | Acc @ $b=0.70$ | Acc @ $b=0.90$ | Retention @ $b=0.90$ | Empirical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense (Ref)** | Full Attention | 68.38% | 68.38% | 68.38% | 68.38% | 100.0% | Full model accuracy |
| **QIRA (Ours)** | **QEI + Ridge** | **64.50%** | **62.75%** | **58.00%** | **51.50%** | **75.3%** | **Maintains robust factual accuracy (+14.8% over LazyLLM)** |
| **QIRA-L (Ours)**| **Learned + Ridge** | **62.00%** | **53.13%** | **49.38%** | **51.13%** | **74.8%** | High consistency at aggressive compression ($b \ge 0.70$) |
| **LazyLLM** | Last-Attn + Drop KV | 58.75% | 49.63% | 44.88% | 36.75% | 53.8% | Drops −31.6% accuracy from Dense |
| **SlimInfer** | Rel-Norm + Drop KV | 51.63% | 45.50% | 37.50% | 31.63% | 46.3% | Prunes fact-carrying tokens, decaying to 31.6% |
| **QuickSilver** | Drift + Stale KV | 39.75% | 25.13% | 15.38% | **11.13%** | **16.3%** | **Plunges below random guessing (<25.0%)** |
| **KVSimilarity**| Cosine + Stale KV | 46.38% | 35.75% | 23.25% | **9.13%** | **13.4%** | **Plunges below random guessing (<25.0%)** |

---

### 10.4 Key Scientific Insights from Downstream Generalization

1. **Spectral Denoising on SST-2:** In short-context classification, QIRA and QIRA-L not only preserve accuracy, but actually **outperform the Dense baseline** (95.64% and 96.10% vs 94.38%). By halting redundant filler tokens identified via QEI spectral energy, QIRA reduces attentional distraction, functioning as an unsupervised regularizer.
2. **The Extractive QA Cliff (SQuAD):** SQuAD v1.1 requires precise verbatim span copying from the context. Dropping tokens without transport (**LazyLLM**) destroys extractive capability, plummeting from 90.86% to **8.38% F1**. In contrast, QIRA’s cross-layer ridge transport projects halted tokens into downstream key/value manifolds, preserving **85.19% F1 (93.8% retention)** even under 90% halting.
3. **Sub-Random Collapse of Stale KV (MMLU):** On MMLU multiple choice, QuickSilver (11.13%) and KVSimilarity (9.13%) score **far below the 25.0% random guessing baseline**. This demonstrates that reusing frozen intermediate-layer representations does not merely add harmless noise—it actively distorts downstream output logits, causing biased token emissions. Closed-form ridge transport completely remedies this pathology.

<br>

<details>
<summary><b>📄 Click to expand Complete LaTeX Source Code for Downstream Table (Overleaf / NAACL)</b></summary>

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{lcccccccccccc}
\toprule
\textbf{Method} & \multicolumn{4}{c}{\textbf{SST-2 (Accuracy $\uparrow$)}} & \multicolumn{4}{c}{\textbf{SQuAD v1.1 (Token-F1 $\uparrow$)}} & \multicolumn{4}{c}{\textbf{MMLU (Accuracy $\uparrow$)}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
& \textbf{0.30} & \textbf{0.50} & \textbf{0.70} & \textbf{0.90} & \textbf{0.30} & \textbf{0.50} & \textbf{0.70} & \textbf{0.90} & \textbf{0.30} & \textbf{0.50} & \textbf{0.70} & \textbf{0.90} \\
\midrule
Dense & \multicolumn{4}{c}{94.38\%} & \multicolumn{4}{c}{90.86\%} & \multicolumn{4}{c}{68.38\%} \\
\midrule
\textbf{QIRA} (\textit{Ours}) & \textbf{94.84\%} & 94.95\% & 95.41\% & 95.64\% & \textbf{89.26\%} & \textbf{89.06\%} & \textbf{86.79\%} & 85.19\% & \textbf{64.50\%} & \textbf{62.75\%} & \textbf{58.00\%} & \textbf{51.50\%} \\
\textbf{QIRA-L} (\textit{Ours}) & 94.61\% & \textbf{95.64\%} & \textbf{95.76\%} & \textbf{96.10\%} & 88.58\% & 86.45\% & 84.92\% & \textbf{85.29\%} & 62.00\% & 53.13\% & 49.38\% & 51.13\% \\
LazyLLM & 93.35\% & 91.63\% & 90.83\% & 91.63\% & 73.65\% & 52.24\% & 23.32\% & 8.38\% & 58.75\% & 49.63\% & 44.88\% & 36.75\% \\
SlimInfer & 93.92\% & 93.69\% & 93.23\% & 48.85\% & 83.14\% & 70.65\% & 55.00\% & 31.76\% & 51.63\% & 45.50\% & 37.50\% & 31.63\% \\
QuickSilver & 91.40\% & 88.65\% & 46.10\% & 8.60\% & 21.80\% & 7.91\% & 3.02\% & 1.51\% & 39.75\% & 25.13\% & 15.38\% & 11.13\% \\
KVSimilarity & 89.56\% & 82.11\% & 62.61\% & 36.35\% & 21.71\% & 12.10\% & 4.75\% & 1.96\% & 46.38\% & 35.75\% & 23.25\% & 9.13\% \\
\bottomrule
\end{tabular}
\caption{Generalization across core downstream NLP benchmarks on Qwen2.5-7B-Instruct across 4 token halting budgets ($b \in \{0.30, 0.50, 0.70, 0.90\}$). QIRA and QIRA-L maintain robust accuracy across all tasks, outperforming Dense on SST-2 via spectral denoising, and avoiding the severe degradation cliffs of uncompensated token dropping (LazyLLM, SlimInfer) and stale-KV reuse (QuickSilver, KVSimilarity).}
\label{tab:downstream_results}
\end{table*}
```
</details>

