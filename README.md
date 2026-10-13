# Frozen but Not Forgotten: Closed-Form Cross-Layer K/V Transport for Intermediate-Layer Token Halting in LLM Prefill

> **Anonymous Repository** accompanying a submission to the **ACL Rolling Review (ARR)**.

---

## Overview

In transformer prefill, intermediate-layer token halting reduces redundant computation by freezing selected prompt tokens past a decision layer $d$. However, existing methods either drop halted tokens entirely or reuse stale key/value states, inducing representation drift in deeper layers.

**QIRA** resolves this through **closed-form cross-layer K/V transport**:
- **Zero backbone modification:** The underlying model weights remain frozen.
- **Closed-form transport:** Keys and values of halted tokens are mapped to deeper layers via layer-specific Ridge regression transport matrices fitted offline on disjoint calibration sequences.
- **Fast calibration:** Calibration finishes in 1-2 minutes on a single GPU.
- **Variants:**
  - **QIRA-L:** Learns a lightweight linear selector over ten decision-layer geometric cues.
  - **QIRA:** Uses a closed-form Quantum-Inspired Entropy of Information (QEI) spectral selector with the same ridge transport.

---

## 1. Environment Setup

Tested on modern NVIDIA GPUs (e.g., RTX 6000 Ada / A100 / Blackwell architectures) with CUDA 12.8+ in `bfloat16` precision.

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2. Install PyTorch with modern CUDA support (CUDA 12.8 / 13.0)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install -r requirements.txt
```

### Models

The benchmark suite supports:
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct` (or public mirrors such as `NousResearch/Meta-Llama-3.1-8B-Instruct`, which require no gated Hugging Face token).

*(Optional)* If weights and datasets are already downloaded and cached locally:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 2. Reproducing Paper Experiments

Every evaluation measurement is recorded under `results/<model>/`. Re-running commands automatically resumes any interrupted runs without duplicate computation.

```bash
# Step 0: Sanity check (verifies engine matches native Hugging Face forward pass)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset quick \
    --experiments selftest calibrate

# Step 1: Full evaluation suite at paper scale
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset paper
python run_benchmark.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset paper

# Step 2: Regenerate publication tables and figures (CPU-only, no GPU needed)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --experiments report
```

Check `results/<model>/selftest.json` (`"passed": true`) to confirm numerical parity before executing long evaluations on new models.

---

> ### Note on Raw Evaluation Traces (`main.jsonl`)
> Due to standard anonymous code repository hosting constraints (specifically the strict **100 MB per-file upload limit**), the raw per-token sample generation dumps (`results/<model>/main.jsonl`, which span **120 MB to 220 MB per model**) are excluded from this release.
> 
> **Everything required to inspect and verify the findings is included:**
> - All audited CSV tables (`tables/*.csv`)
> - Vector and high-resolution figures (`figures/`)
> - Full Markdown evaluation reports (`report.md`)
> - Self-test validation records (`selftest.json`)
> - Intermediate benchmarks (`Dense.jsonl`, `latency.jsonl`, `scaling.jsonl`)
> 
> Reviewers can regenerate the complete uncompressed `main.jsonl` traces from scratch at any time by executing the commands in Step 1 above.

---

### Command-Line Arguments

| Flag | Description |
| :--- | :--- |
| `--experiments ...` | Select subset: `selftest calibrate main transport selector calib_transfer operating latency scaling report` (default: all) |
| `--methods ...` | Select subset of evaluation baselines (see Section 4) |
| `--budgets ...` | Override halting reduction ratios (default: `0.3 0.5 0.7 0.9`) |
| `--attn-backend` | `auto` (FlexAttention if available, otherwise chunked masked attention), `flex`, or `mask` |
| `--max-samples N` | Cap sample count per task (useful for rapid sanity checks) |
| `--recalibrate` | Force refitting the transport maps even if cached weights exist |

---

## 3. Experimental Suite Breakdown

| Experiment | Measured Quantities |
| :--- | :--- |
| `selftest` | Numerical parity against Hugging Face next-token distributions; oracle K/V sanity checks. |
| `calibrate` | Closed-form Ridge transport and selector fitting; held-out K/V reconstruction NMSE across policies. |
| `main` | Full benchmark comparison at matched reduction budgets across LongBench, SQuAD, MMLU, and SST-2 (Retention, KL divergence, Top-1 accuracy, theoretical FLOPs). |
| `transport` | Disentangles transport policies under identical halting sets (Stale, Drop, Reprojection, Ridge variants, Residual MLP, Oracle). |
| `selector` | Disentangles selector criteria under identical transport maps. |
| `latency` | Wall-clock prefill speedup (isolated warm-up passes, median over repeated runs). |
| `scaling` | Long-context needle retrieval scaling (2K to 32K tokens): prefill latency, memory footprint, and retrieval accuracy. |
| `report` | Compiles audited CSV tables (`tables/*.csv`), visual plots, and Markdown summary reports (`report.md`). |

---

## 4. Evaluated Methods

All methods are implemented within the identical inference engine using the same decision layer $d = \lfloor L/2 \rfloor$, preserved boundary tokens (first and last 16 tokens), and matched reduction budgets:

| Method | Token Selector | Downstream K/V State of Halted Tokens |
| :--- | :--- | :--- |
| **QIRA-L** | Closed-form learned linear selector | Ridge transport |
| **QIRA** | Quantum-Inspired Entropy of Information (QEI) | Ridge transport |
| **QuickSilver-style** | Minimum layer drift | Stale cache reuse |
| **LazyLLM-style** | Minimum last-token attention weight | Dropped from cache |
| **SlimInfer-style** | Minimum relative hidden norm | Dropped from cache |
| **KV-Similarity** | Maximum cosine similarity to preceding token | Stale cache reuse |
| **TokenSparse** | Per-head sparse attention baseline | N/A (all tokens updated) |
| **DepthWeave-style** | Attention received from probe queries | Low-rank projected transport |
| **QuickSilver + T** | QuickSilver drift selector | Ridge transport |
| **LazyLLM + T** | LazyLLM attention selector | Ridge transport |

*Note: Baselines re-implement each technique's core halting logic within a unified framework to ensure fair, strictly matched comparisons under identical hardware conditions.*

---

## 5. Repository Structure

```text
.
|-- run_benchmark.py       # Unified CLI execution driver
|-- qira/                  # Core library
|   |-- engine.py          # Prefill engine supporting dynamic halting and transported K/V
|   |-- transport.py       # Closed-form ridge transport (calibration and inference)
|   |-- selectors.py       # QIRA-L, QIRA, and baseline token selection policies
|   |-- calibration.py     # Disjoint calibration loader and closed-form solvers
|   |-- experiments.py     # Benchmark protocol definitions
|   |-- tasks.py           # Evaluation dataset adapters
|   |-- scoring.py         # Downstream evaluation metrics (Rouge-L, Exact Match, etc.)
|   \-- report.py          # Automated generation of tables, plots, and markdown summaries
|-- requirements.txt       # Python dependencies
\-- results/<model>/       # Experimental outputs (CSV tables, plots, selftest.json, summary reports)
```

---

## 6. Data and Licensing

- Model weights and benchmark datasets are retrieved from their original upstream providers and are subject to their respective licenses (Qwen2.5: Apache-2.0; LLaMA-3.1: Llama 3.1 Community License; LongBench: MIT; SQuAD v1.1: CC BY-SA 4.0; MMLU: MIT; SST-2: Stanford Sentiment Treebank terms).
- This repository distributes no proprietary weights or proprietary dataset splits.
