# QIRA v8 — QEI-guided token halting with cross-layer K/V transport

**Claim under test.** Intermediate-layer token halting is limited by *cross-layer K/V
representation mismatch*: a token halted at layer *d* still has to serve as a key/value
at every later layer, but its frozen hidden state lives in layer *d*'s representation
space. QIRA (i) decides **which** tokens can halt with a spectral redundancy index (QEI)
and (ii) **transports** the halted tokens' K/V into each downstream layer's space with a
closed-form, per-layer / per-KV-head ridge map.

QIRA is **backbone-frozen and inference-time adaptive**. No model weight is changed. The
transport maps are calibrated once, offline and in closed form, on a calibration pool that
is disjoint from every evaluation set. Do not call it "training-free".

---

## 1. Setup (RTX PRO 6000 Blackwell, 96 GB)

Blackwell (sm_120) **requires PyTorch built for CUDA ≥ 12.8**. Older cu121/cu124 wheels
fail with "no kernel image is available".

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export HF_TOKEN=hf_...            # needed for meta-llama models (accept the licence on the HF website first)
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Use `tmux` for long runs. Every measurement is appended to `results/<model>/<experiment>.jsonl`.
If a run stops, re-running the **same command** resumes where it left off.

## 2a. Updating the existing Qwen2.5-7B results to v8.1 (incremental)

Copy your existing `results/Qwen2.5-7B-Instruct/` folder into the new code directory, then run:

```bash
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset paper --reset-experiments scaling
```

* Calibration is refitted once (a few minutes). The ridge maps are identical to before, plus the new learned selectors.
* `main` only runs the new methods: **QIRA-L**, **QuickSilver+T** and **LazyLLM+T**. Existing rows are reused.
* `selector` only runs `qei_fit` and `learned`. `latency` is new. `scaling` is re-measured with the FlexAttention kernel.
* To save time, restrict with e.g. `--methods Dense QIRA QIRA-L QuickSilver+T LazyLLM+T`.

## 2b. Llama / Mistral access (the gated-repo error)

`meta-llama/*` and recent `mistralai/*` repos are gated. Either accept the licence on the model page and run
`huggingface-cli login` (or `export HF_TOKEN=hf_...` with a token that has been granted access), or use an
ungated mirror of the same weights:

```bash
python run_benchmark.py --model unsloth/Llama-3.1-8B-Instruct --preset paper        # same weights as meta-llama
python run_benchmark.py --model unsloth/mistral-7b-instruct-v0.3 --preset paper
python run_benchmark.py --model Qwen/Qwen3-8B --preset paper                         # QK-norm; thinking mode disabled
```

Always check `selftest.json` ("passed": true) for a new architecture before trusting its numbers.

## 2. Run order (recommended)

```bash
# 0) 5-minute sanity check: engine == HF forward, sparse path exact with oracle K/V, calibration works
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset quick \
    --experiments selftest calibrate --tasks idbench

# 1) quick end-to-end study (all experiments, small sample counts), roughly 1-2 h for a 7B model
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset quick

# 2) paper-scale study, roughly 8-12 h for a 7B model
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset paper

# 3) other backbones (Llama-3.1-8B supports the 32K scaling points; Mistral v0.2 is fine up to 32K)
python run_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct --preset paper
python run_benchmark.py --model mistralai/Mistral-7B-Instruct-v0.2 --preset paper

# rebuild tables / figures only (no GPU needed)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --experiments report
```

Useful flags:

| Flag | Effect |
|---|---|
| `--experiments ...` | any of `selftest calibrate main transport selector calib_transfer operating latency scaling report` (default `all`) |
| `--reset-experiments X Y` | delete those experiments' result files first (e.g. `scaling latency`) |
| `--attn-backend` | `auto` (FlexAttention on CUDA if it validates, else masked SDPA), `flex`, or `mask` |
| `--tasks ...` | subset of `idbench longbench squad mmlu gsm8k humaneval` |
| `--max-samples N` | cap samples per task (debugging) |
| `--budgets ...` | override the main-experiment halting budgets |
| `--calib-source` | `natural` (default), `idbench` (the v7 paper pool) or `code` |
| `--recalibrate` | refit the transport even if a matching calibration is cached |
| `--no-mlp` | skip the nonlinear residual-MLP transport variant |
| `--scaling-lengths 2048 4096 ...` | override the context-scaling lengths |
| `--attn-chunk 4096` | query chunk size of the masked sparse attention (lower it if you hit OOM at 32K) |
| `--force-restart` | delete this model's result files first |

**Send back the whole `results/<model>/` folder.** It contains `report.md`, `tables/*.csv`,
`figures/*.png`, the raw `*.jsonl` files, `selftest*.json` and `calibration/`.

## 3. What each experiment answers

| Experiment | Question | Output |
|---|---|---|
| `selftest` | Does the manual engine reproduce the HF model exactly? With the true K/V for halted tokens (`oracle`), does the sparse path give the dense result? | `selftest.json` (the run aborts if this fails) |
| `calibrate` | How large is the cross-layer mismatch? How much does each KV policy remove? | Fig. 1, `transport_diag.csv`, learned QEI weights, threshold quantiles, selector/transportability correlation |
| `main` | Matched token budget: Dense vs QIRA vs QuickSilver vs LazyLLM vs SlimInfer vs KVSimilarity | Table "matched budget", Table "matched quality" (99/97/95/90/85 %), Fig. 3 |
| `transport` | **Killer experiment.** Same QEI halting set, different KV policy: stale, drop, reproject, ridge_plain, ridge_layer, ridge, mlp, oracle | Fig. 4 |
| `selector` | Is QEI more than cosine or norm-delta? Tests drop-one variants, equal weights, learned weights, exact spectrum, cosine, norm-delta, drift, last-attention, rel-norm and random, all with the same transport, plus Jaccard overlap of the halt sets | selector table |
| `calib_transfer` | Is calibration lightweight and transferable? Compares transport calibrated on synthetic IDBench, natural LongBench text and code | transfer table |
| `operating` | Adaptive threshold mode at Conservative / Balanced / Aggressive operating points | Fig. 2 stars |
| `latency` | Clean wall-clock prefill speed-up on LongBench prompts ≥ 2K tokens (warm-up + median of repeats, no generation) | `latency.csv`, used for Fig. 3 and the matched-quality speed-ups |
| `scaling` | Needle retrieval at 2K to 32K: prefill latency (median of repeats after warm-up), TTFT, peak memory, prefill tokens/s | Fig. 5, `scaling.csv` |

## 4. Methods (all run in the same engine with the same decision layer d = L/2)

| Method | Selector (higher score = halt) | K/V of halted tokens at layers ≥ d |
|---|---|---|
| **QIRA** | QEI = 0.35·purity + 0.25/rank + 0.20·gap + 0.20·stability | **ridge transport** from the frozen state |
| **QIRA-L** | transport-aware learned selector: ridge regression of −log(halting harm) on 10 cheap features | ridge transport |
| **TokenSparse (Jo et al., 2026)** | window query attention caching with adaptive prefix windowing | dynamic per-head sparse attention (no transport) |
| **DepthWeave (Cordoba et al., 2026)** | multi-head attention sensitivity probe router | cross-layer shared SVD residual basis factorization ($B B^\top \Delta W_l$) |
| QuickSilver+T / LazyLLM+T | baseline selector | ridge transport (decomposition) |
| QuickSilver-style | low representation drift between layers d−1 and d | stale (KV skipping) |
| LazyLLM-style | low attention from the last token at layer d | dropped (single-stage pruning, no revival) |
| SlimInfer-style | low relative hidden-state norm | dropped |
| KVSimilarity | high cosine similarity with the previous token | stale |

The baselines are re-implementations of each method's *halting criterion and KV handling*
inside a shared engine. They are not the authors' code (no fusion, quantisation or revival).
Say exactly this in the paper.

**Matched budgets.** In `budget` mode every method halts exactly
`round(b · eligible)` tokens per prompt. Eligible tokens exclude the first 16 (attention sinks
and the QEI window) and the last 16 (question and generation prompt). In `threshold` mode,
τ(r) is the (1−r)-quantile of each selector's calibration scores, so r is the expected halt
rate **for every method**. In v7 the direction was inverted between QIRA and the baselines.

## 5. Metrics (reported separately, never mixed)

* `halt_frac`: token reduction (halted / prompt tokens). `token_layer_reduction`: skipped (token, layer) pairs.
* `flop_reduction`: analytic prefill FLOPs including causal attention and the transport overhead.
* `speedup`: wall-clock prefill time of the **Dense engine** divided by the method's, on the same
  sample, CUDA-synchronised. The HF forward time is logged too (`hf_prefill_s`).
  Short prompts are dominated by fixed overhead, so quote `speedup_long_median` (≥ 2K tokens) and the scaling experiment.
* Quality: one primary metric per task (IDBench accuracy, SQuAD F1, MMLU accuracy, GSM8K accuracy,
  HumanEval pass@1, LongBench qa-F1). Retention = macro average over tasks of method / Dense, with bootstrap 95 % CI.
* Low-variance fidelity: `kl_last` (next-token KL to Dense), `top1_agree`, `ans_nll_delta`
  (teacher-forced NLL of the reference answer), `fidelity` (ROUGE-L to the Dense output).

## 6. Code map

```
run_benchmark.py      CLI, selftest, orchestration
qira/config.py        QIRAConfig, method registry, presets (quick / paper)
qira/models.py        model loading, geometry, EOS handling, sliding-window guard
qira/engine.py        prefill with halting, KV policies, own KV cache, decode, teacher-forced NLL
qira/selectors.py     QEI (vectorised GPU) + baseline selectors, budget/threshold halting
qira/transport.py     sufficient statistics, ridge fits with held-out alpha, DepthWeave SVD basis, residual MLP
qira/token_sparse.py  TokenSparseAttentionWrapper (vectorized PyTorch & Triton dynamic token sparsification)
qira/tsa_kernel.py    Triton kernel for Token-Sparse window attention caching
qira/calibration.py   calibration pass: transport, thresholds, learned QEI weights, diagnostics
qira/tasks.py         prompts (chat template), middle truncation, LongBench, IDBench, needle, calibration pools
qira/scoring.py       task metrics, analytic FLOPs
qira/experiments.py   experiment drivers (resumable)
qira/report.py        tables, matched-quality interpolation, figures, report.md
```

Supported backbones: LLaMA-2/3.x, Qwen2/2.5/3 and Mistral (RoPE, RMSNorm, GQA, optional QK-norm).
`selftest` refuses to continue if the engine does not reproduce the model.
Sliding-window attention is not re-implemented. Prompts longer than the window raise an error.

See `CHANGES.md` for every bug fixed, `RESULTS_QWEN7B_v8.md` for the analysis of the first Qwen2.5-7B run,
and `PAPER_NOTES.md` for the paper story.
The v7 notebooks are kept for provenance only.
