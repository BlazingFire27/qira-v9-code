# Changes from v7 (Benchmark_2 port) to v8

## Bugs fixed

| # | v7 behaviour | Effect | v8 |
|---|---|---|---|
| 1 | Halted tokens kept their **layer d−1 K/V for every later layer**. No transport existed in the code, although the paper describes ridge transport. | Every v7 "QIRA" number measured stale-KV halting | Explicit KV policies: `stale`, `drop`, `reproject`, `ridge*`, `mlp`, `oracle` |
| 2 | Threshold direction inverted: QIRA halted the **top 1−τ** of tokens, LazyLLM/SlimInfer the **bottom τ** | At τ = 0.80 QIRA halted ~20 % of tokens and the baselines ~80 %. Comparisons were not budget-matched | Exact per-prompt budgets for all methods. Threshold mode uses τ(r) = (1−r)-quantile for every selector |
| 3 | `tokenizer(..., truncation=True, max_length=2048)` cut prompts from the right | Every LongBench prompt longer than 2K lost its question and "Answer:" | Middle truncation (LongBench protocol). The default context is 8K in the paper preset |
| 4 | HumanEval executed `completion + tests` without the prompt / signature | pass@1 was near 0 by construction | `prompt + completion`, or the full function extracted from a code block |
| 5 | Fusion copied the representative's **already-rotated** K to another position | Wrong RoPE phase for fused tokens | Fusion removed. It confounded halting and was buggy |
| 6 | Cheap-gate tokens were force-halted after thresholding | The threshold did not control the halt rate | Cheap-gate tokens get maximal spectral components but go through the same budget/threshold |
| 7 | Prefill attention was a naive fp32 n×n matmul | O(n²) memory: OOM far below 32K; no real speed-up possible | SDPA (flash/causal) for dense layers; chunked masked SDPA for sparse layers |
| 8 | Only the prefill was timed, without CUDA synchronisation inside `generate`; no warm-up | Unreliable TTFT | Synchronised timing, global warm-up, median of repeats in the scaling experiment |
| 9 | FLOP estimator counted one d×d projection for Q+O and full (non-causal) attention | Wrong reduction percentages | Q, K, V, O, gated MLP and causal pair counts, plus transport overhead |
| 10 | 1 sample each for GSM8K / MMLU / SQuAD / HumanEval | Statistically meaningless | 30 / 100 / 50 / 20 samples (paper preset), LongBench 40 over 4 subsets |
| 11 | `quality` = mean of all score fields (F1 averaged with EM) | Mixed metrics | One primary metric per task |
| 12 | SQuAD v2 gold `[""]` for unanswerable questions while the prompt asked for "unanswerable" | Correct refusals scored 0 | SQuAD v1.1 (always answerable) |
| 13 | Decoding relied on HF `DynamicCache` construction, which changes across transformers versions | Fragile; `drop` was impossible | The engine owns a pre-allocated KV cache; decode honours every KV policy |
| 14 | No check that the manual layer reproduces the model | Silent errors possible (bias, QK-norm, scaling, RoPE scaling) | `selftest`: engine vs HF KL, decode match, NLL match, oracle-KV exactness |
| 15 | Raw prompts on instruct models | Low Dense quality, noisy comparisons | Chat template by default (`--no-chat-template` to disable) |
| 16 | Per-token Python loop with CPU transfers for QEI | Slow at long context, included in TTFT | Vectorised GPU QEI (same formulas) |

## Method changes
* Cross-layer K/V transport (closed-form ridge, per layer and per KV head, shrinkage towards the
  layer's own projections, held-out regularisation). Variants: `ridge_layer`, `ridge_plain`, residual MLP.
* QEI weights can be learned (non-negative least squares against token transportability). Sensitivity variants included.
* Larger and more varied calibration pools (natural, IDBench, code); calibration-transfer experiment.
* Terminology: "backbone-frozen, inference-time adaptive, closed-form transport calibration".

## v8.1 (after the first Qwen2.5-7B paper run; see RESULTS_QWEN7B_v8.md)
* **Speed.** FlexAttention block-sparse kernel for transported policies. It is validated against masked SDPA
  at start-up and switches to masked SDPA automatically if it ever fails. `drop` now uses causal flash attention
  on the compacted active sequence. The attention mode is logged per row (`attn_mode`).
* **New `latency` experiment.** Long prompts only, warm-up plus median of repeats, no generation.
  Its speed-ups replace the single-shot timings taken inside generation runs in Fig. 3 and in the matched-quality table.
* **Transport-aware learned selectors.** `learned` (method QIRA-L) and `qei_fit` use closed-form ridge
  regression on per-token halting harm (attention received × ridge transport error). The broken
  NNLS `qei_learned` is removed.
* **Decomposition methods.** QuickSilver+T and LazyLLM+T (baseline selector + QIRA transport), with a decomposition table.
* **Report.** `n` shown everywhere. Under-sampled cells (left-overs of a quick run) are dropped. Selector alignment
  is reported against halting harm and against transportability.
* `ridge_gated` removed from the default ablation (calibrated β was 0).
* `--reset-experiments`, `--attn-backend`. Qwen3 thinking mode disabled in the chat template.
* Result keys are unchanged, so existing v8 results resume and only new rows are computed.

## v9.0 (Full 16-Task LongBench & Token-Sparse Attention Baseline)
* **Token-Sparse Attention (TSA).** Integrated official Token-Sparse Attention (Jo et al., ICML 2026)
  into `qira.token_sparse` and `qira.engine`. Added `TokenSparse` to `METHOD_REGISTRY`, `MAIN_METHODS`,
  and `LATENCY_METHODS`.
* **Decoupled Dense Reference Precomputation & Caching.** Added `--dense-only` to evaluate and serialize
  Dense reference outputs into `dense_references.pt` and `Dense.jsonl` once. Added `--load-dense` so all
  downstream baseline instances load pre-computed references in ~2 seconds with zero redundant Dense passes.
* **Full 16-Task LongBench Support.** Added `--longbench-all` evaluating all 16 official LongBench tasks
  (~3,750 samples) with official middle-truncation prompt templates and task-specific QA/Rouge metrics.
* **Turnkey Multi-GPU Orchestration.** Added `scripts/run_phase1_dense.sh`, `scripts/run_phase2_gpu1.sh`,
  `scripts/run_phase2_gpu2.sh`, `scripts/run_phase2_gpu3.sh`, and `scripts/run_phase3_merge.sh`.

