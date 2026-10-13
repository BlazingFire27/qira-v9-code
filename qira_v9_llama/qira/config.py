"""
QIRA v8 configuration.

QIRA = QEI-guided intermediate-layer token halting + cross-layer K/V transport.

Terminology used throughout the code and reports:
  * backbone-frozen: the LLM weights are never modified or fine-tuned;
  * inference-time adaptive: the halting set is decided per prompt at run time;
  * closed-form transport calibration: the per-layer / per-KV-head ridge
    transport maps are fitted once, offline, in closed form on a small
    calibration pool that is disjoint from every evaluation set.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

PROTOCOL_VERSION = "QIRA-v8-transport"   # unchanged so existing results resume
CODE_VERSION = "9.0"


@dataclass
class QIRAConfig:
    # ---------------- decision point ----------------
    decision_layer_fraction: float = 0.5   # halting decided at the input of layer floor(L * frac)
    protect_prefix: int = 16               # first tokens (attention sinks / QEI warm-up window) never halt
    protect_suffix: int = 16               # last tokens (question / generation prompt) never halt

    # ---------------- QEI (spectral redundancy index) ----------------
    window_w: int = 16
    projection_dim: int = 8
    projection_seed: int = 1234
    qei_weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)  # purity, 1/rank, gap, stability
    spectral_mode: str = "lite"            # "lite" (trace / power-iteration proxies) or "exact" (batched eigvalsh)
    power_iterations: int = 1
    # Stage-0 cheap local-redundancy gate (tokens passing it receive maximal spectral components)
    cheap_gate_enabled: bool = True
    cheap_cosine_threshold: float = 0.998
    cheap_norm_delta_threshold: float = 0.02
    # Hysteresis (threshold mode only): once halted, stay halted while score >= quantile(1 - r - band)
    hysteresis_band: float = 0.05

    # ---------------- transport calibration ----------------
    ridge_alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0)
    val_every: int = 5                     # every k-th calibration prompt is held out for alpha selection
    mlp_hidden: int = 512
    mlp_steps: int = 400
    mlp_max_tokens: int = 32768
    mlp_lr: float = 1e-3
    depthweave_rank: int = 64

    # ---------------- engine ----------------
    attn_query_chunk: int = 4096           # query chunk for masked sparse attention


# Methods = (selector, KV policy for halted tokens).  "Dense" has no halting.
# Native KV policies follow each method's published design as closely as the
# shared engine allows (see README "Baselines").
METHOD_REGISTRY: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "Dense":        (None, None),
    "QIRA":         ("qei", "ridge"),          # ours: QEI selection + closed-form per-layer/per-head transport
    "QuickSilver":  ("drift", "stale"),        # representation-drift halting + KV skipping
    "LazyLLM":      ("lastattn", "drop"),      # last-token attention pruning (single stage, no revival)
    "SlimInfer":    ("relnorm", "drop"),       # hidden-state-norm pruning
    "KVSimilarity": ("cosine", "stale"),       # adjacent-token similarity
    # v8.1 additions
    "QIRA-L":       ("learned", "ridge"),      # ours: transport-aware learned selector + ridge transport
    "QuickSilver+T": ("drift", "ridge"),       # baseline selector with QIRA transport plugged in
    "LazyLLM+T":    ("lastattn", "ridge"),
    # v9 additions: Token-Sparse Attention (Jo et al.)
    "TokenSparse":  ("tsa", "sparse_attn"),    # per-head dynamic token sparse attention
    # DepthWeave-KV (Cordoba et al., 2026): Token-Adaptive Cross-Layer Residual Factorization
    "DepthWeave":   ("depth_router", "depthweave"),
}
MAIN_METHODS = ["Dense", "QIRA", "QIRA-L", "QuickSilver", "LazyLLM", "SlimInfer", "KVSimilarity",
                "QuickSilver+T", "LazyLLM+T", "TokenSparse", "DepthWeave"]
LATENCY_METHODS = ["QIRA", "QIRA-L", "QuickSilver", "LazyLLM", "SlimInfer", "TokenSparse", "DepthWeave"]

SELECTORS = [
    "qei", "qei_equal", "qei_no_purity", "qei_no_rank", "qei_no_gap", "qei_no_stability",
    "qei_exact", "qei_fit", "learned", "cosine", "normdelta", "drift", "lastattn", "relnorm", "random",
    "tsa", "depth_router",
]
TRANSPORTS = ["stale", "drop", "reproject", "ridge_plain", "ridge_layer", "ridge", "ridge_gated", "mlp", "oracle", "sparse_attn", "depthweave"]
CALIB_SOURCES = ["natural", "idbench", "code"]

# Generation budget per task (new tokens).
MAX_NEW_TOKENS = dict(idbench=16, squad=32, mmlu=16, gsm8k=320, humaneval=384, longbench=64, needle=16, sst2=8)

TASKS = ["idbench", "longbench", "sst2", "gsm8k", "squad", "mmlu", "humaneval"]
LONGBENCH_EVAL_SUBSETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa"]
LONGBENCH_ALL_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum",
    "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]
LONGBENCH_CALIB_NATURAL = ["musique", "2wikimqa", "gov_report", "multi_news"]
LONGBENCH_CALIB_CODE = ["lcc", "repobench-p"]

IDBENCH_REDUNDANCY = [0, 25, 50, 75, 90]
IDBENCH_SIGNAL_UNITS = 12


@dataclass
class Preset:
    name: str
    samples: Dict[str, int]
    idbench_eval_per_density: int
    max_context: int
    main_budgets: List[float]
    transport_budgets: List[float]
    selector_budgets: List[float]
    transfer_budgets: List[float]
    operating_points: Dict[str, float]
    transport_tasks: List[str]
    selector_tasks: List[str]
    transfer_tasks: List[str]
    scaling_lengths: List[int]
    scaling_samples: int
    scaling_budgets: List[float]
    timing_repeats: int
    calib_prompts: Dict[str, int]
    calib_max_tokens: int
    latency_samples: int = 20


PRESETS: Dict[str, Preset] = {
    "quick": Preset(
        name="quick",
        samples=dict(longbench=8, squad=800, mmlu=800, gsm8k=1319, humaneval=5, sst2=872),
        idbench_eval_per_density=2,
        max_context=4096,
        main_budgets=[0.3, 0.5, 0.7, 0.9],
        transport_budgets=[0.3, 0.6, 0.9],
        selector_budgets=[0.5, 0.8],
        transfer_budgets=[0.5, 0.8],
        operating_points=dict(Conservative=0.3, Balanced=0.5, Aggressive=0.7),
        transport_tasks=["idbench", "squad", "longbench"],
        selector_tasks=["idbench", "squad", "longbench"],
        transfer_tasks=["idbench", "squad", "longbench"],
        scaling_lengths=[2048, 4096, 8192],
        scaling_samples=1,
        scaling_budgets=[0.5, 0.8],
        timing_repeats=3,
        calib_prompts=dict(natural=16, idbench=20, code=8),
        calib_max_tokens=2048,
        latency_samples=8,
    ),
    "paper": Preset(
        name="paper",
        samples=dict(longbench=40, squad=800, mmlu=800, gsm8k=1319, humaneval=20, sst2=872),
        idbench_eval_per_density=6,
        max_context=8192,
        main_budgets=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
        transport_budgets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        selector_budgets=[0.5, 0.7, 0.9],
        transfer_budgets=[0.5, 0.7, 0.9],
        operating_points=dict(Conservative=0.3, Balanced=0.5, Aggressive=0.7),
        transport_tasks=["idbench", "squad", "longbench", "mmlu"],
        selector_tasks=["idbench", "squad", "longbench", "mmlu"],
        transfer_tasks=["idbench", "squad", "longbench", "mmlu"],
        scaling_lengths=[2048, 4096, 8192, 16384, 32768],
        scaling_samples=3,
        scaling_budgets=[0.5, 0.7, 0.9],
        timing_repeats=5,
        calib_prompts=dict(natural=48, idbench=20, code=24),
        calib_max_tokens=2048,
        latency_samples=40,
    ),
}


def config_to_dict(cfg: QIRAConfig) -> Dict:
    return asdict(cfg)
