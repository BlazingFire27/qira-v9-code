# Analysis of the v8 paper run on Qwen2.5-7B-Instruct (270 evaluation samples, 6 tasks)

## Headline (supported by the data)
1. **The mismatch problem is real and large.** Held-out K/V NMSE of halted tokens is 2.76 for stale
   reuse (worse than predicting zero), 0.33 for reprojecting the frozen state, and 0.12 for ridge transport.
2. **Transport is what makes halting work (same QEI halting set, Section 5):**

   | halted (budget) | stale | drop | reproject | **ridge** | mlp | oracle |
   |---|---|---|---|---|---|---|
   | 25 % (0.3) | 0.36 | 0.90 | 0.78 | **0.97** | 0.99 | 1.00 |
   | 42 % (0.5) | 0.20 | 0.83 | 0.76 | **0.97** | 0.99 | 0.99 |
   | 59 % (0.7) | 0.20 | 0.67 | 0.74 | **0.93** | 0.94 | 1.00 |
   | 76 % (0.9) | 0.12 | 0.50 | 0.62 | **0.92** | 0.90 | 1.00 |

   Values are quality retention. Next-token KL tells the same story: at 76 % halted, stale 9.15,
   drop 9.54, reproject 2.24, ridge 1.73.
3. **Matched budgets (Section 2).** At 59 % of tokens halted, QIRA retains 0.90 [0.85, 0.96] of Dense
   quality. LazyLLM retains 0.45, SlimInfer 0.58, QuickSilver 0.15 and KVSimilarity 0.18. QIRA is the
   only method whose GSM8K, HumanEval and SQuAD quality survives budgets ≥ 0.5 (`main_per_task.csv`).
4. **Matched quality (Section 3).** At 90 % retention QIRA removes 35 % of prefill FLOPs (73 % of tokens
   halted). LazyLLM removes 13 % and SlimInfer 14 %. QuickSilver and KVSimilarity never reach 85 %.
5. **Calibration transfers.** Natural-text and code pools give equivalent results. The tiny synthetic
   IDBench pool (20 prompts) is clearly worse at high budgets (0.77 vs 0.93 retention at budget 0.9).
   Recommend natural-text calibration and report the IDBench pool as an ablation.
6. **Adaptive threshold mode helps slightly.** QIRA-Aggressive halts 61.5 % with 0.918 retention, while
   the exact budget halts 57.8 % with 0.903.

## Problems found, and what v8.1 changes
| Finding | Cause | v8.1 fix |
|---|---|---|
| Wall-clock speed-up lags the FLOP reduction. At 32K tokens, QIRA@0.5 is **slower** than Dense (0.97×) despite 25 % fewer FLOPs | Sparse layers used chunked SDPA with an explicit mask. That falls off the flash kernel and computes the full masked rectangle | **FlexAttention** block-sparse kernel for transported policies (validated at start-up, automatic runtime fallback). `drop` now runs plain causal flash attention on the compacted sequence. New `latency` experiment (warm-up + median of repeats, long prompts only) |
| `speedup_median` < 1 on short prompts | A 7B prefill on a few hundred tokens is memory-bandwidth bound, so fewer tokens cannot help | Not a bug. Report speed-ups for ≥ 2K-token prompts only (`latency`, `scaling`) |
| QEI does **not** beat simple selectors with the same transport. At budget 0.5, cosine+ridge KL 0.59 vs QEI 1.11 and random 1.03. At budget 0.7, cosine 0.99, drift 1.00, QEI 1.43 | QEI scores local redundancy, but what matters is whether a token's future K/V are predictable *and* how much attention it receives. QEI correlates negatively with transportability (Spearman −0.15) | New **transport-aware learned selector** (`learned`, method **QIRA-L**). It is a closed-form ridge regression of per-token *halting harm* (attention received × ridge transport error, measured in calibration) on 10 cheap decision-layer features. `qei_fit` is the same with only the 4 QEI components |
| `qei_learned` rows were identical to `qei_equal` | All four QEI components correlated negatively with the target, so non-negative least squares returned 0 and fell back to equal weights | Replaced by signed ridge regression (`qei_fit`, `learned`) |
| Selector table showed budget 0.8 with retention 1.000 and KL 0.05 for every selector | Left-over rows from an earlier `--preset quick` run (few samples) in the same result file | Report shows `n` and drops cells with fewer than half the samples of the largest cell |
| Baseline numbers mix two factors (selector and KV policy) | QuickSilver and KVSimilarity collapse mainly because of stale K/V | New main methods **QuickSilver+T** and **LazyLLM+T** (baseline selector + QIRA transport) and a decomposition table (Section 4b) |
| `ridge_gated` identical to `ridge` | Its calibrated logit bias was β = 0, so pure transport is already optimal | Removed from the default ablation (a finding for the paper) |
| Needle accuracy at 32K: QIRA 0.67 (3 samples) | One of three needles lost; too few samples to conclude | Re-run `scaling` (reset) with the Flex kernel |

## Recommended paper framing after this run
The selector result changes the emphasis. The contribution is **cross-layer K/V transport** (and the
diagnosis of the mismatch). The selector is a secondary design choice. Say explicitly that, given transport,
several cheap selectors perform similarly, and that the transport-aware selector (QIRA-L) is the principled
choice *if* it wins in v8.1. If QIRA-L does not beat cosine+ridge, present QEI as one interchangeable
selector, and keep the claim on transport plus the matched-quality efficiency.

## Pre-check of QIRA-L (Qwen2.5-0.5B on CPU, small calibration: 12 prompts × 512 tokens)
* The learned harm predictor reaches held-out Spearman 0.56 (`qei_fit`, QEI components only: 0.25).
* On 8 unseen LongBench prompts (next-token KL, ridge transport): learned beats QEI at budgets 0.5 and 0.8,
  but cosine still has the lowest mean KL. At budget 0.8 learned has the best median and a heavier tail.
  This is too small to conclude. The 7B paper run decides. If cosine+ridge stays best, use it as the
  simple selector and keep the paper's claim on transport.
