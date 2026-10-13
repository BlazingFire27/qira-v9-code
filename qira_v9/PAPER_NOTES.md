# Paper notes: how the experiments map onto the story

## Central claim (write it this way)
> Intermediate-layer token halting is limited by **cross-layer K/V representation mismatch**.
> QIRA makes halting viable by **transporting** halted-token K/V into each downstream layer's space.

Avoid making "X % token reduction" the headline. The headline is the mismatch problem and the
mechanism that fixes it.

## Contributions (three, no more)
1. **Problem.** We identify and quantify cross-layer K/V mismatch as the failure mode of
   intermediate-layer halting. Evidence: Fig. 1. Reusing stale K/V has held-out NMSE ≈ 2, which is
   worse than predicting zero. Reprojecting the frozen state is better but degrades with depth.
2. **Method.** QIRA combines spectral-redundancy (QEI) halting with a closed-form,
   per-layer / per-KV-head ridge transport that shrinks towards the layer's own projections.
   It is backbone-frozen, inference-time adaptive and calibrated in closed form.
3. **Evidence.** Matched-budget and matched-quality comparisons against QuickSilver-, LazyLLM-,
   SlimInfer- and KVSimilarity-style halting. The transport ablation shows the mechanism is necessary.
   Measured prefill speed-ups are reported separately from FLOPs, from 2K to 32K tokens.

## Figure plan (all produced by `report`)
| Fig | File | Message |
|---|---|---|
| 1 | `fig1_transport_mismatch.png` | Why naive halting fails: K/V error per layer for stale / reproject / ridge |
| 2 | `fig2_qira_frontier.png` | QIRA quality–efficiency frontier with Conservative / Balanced / Aggressive points |
| 3 | `fig3_methods.png` | QIRA vs baselines at matched budgets (FLOPs and measured speed-up) |
| 4 | `fig4_transport_ablation.png` | Same halting set, different KV policy, from stale through ridge to the oracle upper bound |
| 5 | `fig5_scaling.png` | Prefill latency and speed-up vs context length |

## Status after the Qwen2.5-7B v8 run (see RESULTS_QWEN7B_v8.md)
* Transport claim: **supported**. Ridge beats stale, drop and reproject at every budget and approaches the oracle.
* Selector claim: **not supported for QEI**. With the same transport, cosine/drift are as good or better.
  v8.1 adds a transport-aware learned selector (QIRA-L). Lead with transport. Treat the selector as secondary
  unless QIRA-L clearly wins.
* Speed: the v8 kernel under-delivered (masked SDPA). Quote v8.1 `latency` / `scaling` numbers only.

## Questions the results must answer honestly (decide the framing after the run)
* **Does ridge beat `drop`?** On Qwen2.5-0.5B smoke tests, simply dropping halted tokens was a
  strong competitor to reprojection (KL 0.01–0.08 vs 0.14–0.55). Ridge closed most of that gap even
  with a tiny calibration pool. `ridge_gated` (calibrated logit bias β on transported keys; β = 0 is
  pure transport, β → −∞ is drop) is the principled interpolation.
  * If `ridge` or `ridge_gated` beats `drop` and `stale` on KL **and** quality: the story stands as written.
  * If `ridge_gated` wins only with a strongly negative β: frame transport as *confidence-weighted*
    K/V recovery. Fig. 1 explains why confidence must decrease with depth.
  * If `drop` wins everywhere: the mismatch claim (Fig. 1) still holds, but the remedy claim must be
    weakened. Do not hide this; reviewers will run `drop`.
* **Is QEI more than cosine?** Check `selector_ablation.csv` (KL and retention against `cosine`,
  `normdelta` and `random`) and the Jaccard overlaps. In the smoke test QEI was barely correlated with
  token transportability. If QEI does not beat `random` + ridge, the contribution is the transport,
  and QEI should be presented as one interchangeable selector.
* **Matched quality.** Quote `main_matched_quality.csv`. If QIRA reaches a target at a larger FLOP
  reduction / speed-up than every baseline, you can say it "dominates the evaluated baselines at
  matched quality". Otherwise say "competitive".
* **Calibration transfer.** If `QIRA[idbench]`, `QIRA[natural]` and `QIRA[code]` are within CI of each
  other, claim "lightweight and transferable calibration".

## Wording rules
* Say "backbone-frozen; closed-form transport calibration", never "training-free".
* Call the baselines "QuickSilver-style" or "LazyLLM-style" re-implementations of their halting criterion and KV
  handling in a shared engine. They are not the official code. QuickSilver's token fusion and
  quantisation, and LazyLLM's token revival, are not included.
* FLOP reduction ≠ wall-clock speed-up ≠ TTFT reduction. Report all three separately.
  Short prompts are overhead-dominated, so use the scaling experiment and `speedup_long_median`.
* No "state of the art". Use "competitive or superior trade-off among the evaluated halting methods".

## Related-work taxonomy (verify every entry against the papers before submission)
| Family | Selection | Halting | Cross-layer K/V transport | Reconsideration | Target |
|---|---|---|---|---|---|
| QuickSilver (2025) | representation drift / entropy | ✓ | — (KV skipping) | — | inference FLOPs |
| LazyLLM (2024) | last-token attention | progressive pruning | — | ✓ (aux cache revival) | long-context TTFT |
| Token Sparse Attention | dynamic attention selection | — | — | ✓ | long-context attention |
| KV-cache compression (e.g. cross-layer sharing) | cache compression | — | sometimes | varies | memory |
| **QIRA** | spectral redundancy (QEI) | ✓ | **✓ closed-form ridge** | future work | prefill |

## Limitations to state
Single decision layer (d = L/2). No token revival. Transport is linear (the MLP variant is reported).
Sliding-window models are not supported. Calibration requires one dense pass over the pool.
Speed-ups depend on the masked-attention kernel available for the GPU.
