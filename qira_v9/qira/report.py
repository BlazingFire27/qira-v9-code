"""
Analysis: matched-budget and matched-quality tables, Figures 1-5, markdown report.

Quality retention for a (method, budget) cell = macro-average over tasks of
    mean(task quality of method) / mean(task quality of Dense)
computed on the identical samples (tasks where Dense scores 0 are skipped).  95% CIs are
bootstrap over samples within each task.  FLOP reduction (analytic), token reduction and
wall-clock prefill speed-up are reported separately and never mixed.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

QUALITY_TARGETS = [0.99, 0.97, 0.95, 0.90, 0.85]
LONG_PROMPT = 2048


def load(out: Path, exp: str) -> pd.DataFrame:
    p = out / f"{exp}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows)


def attach_dense(df: pd.DataFrame) -> pd.DataFrame:
    dense = df[df.method == "Dense"][["sample_id", "quality", "prefill_s"]].rename(
        columns={"quality": "q_dense", "prefill_s": "prefill_dense"}).drop_duplicates("sample_id")
    out = df[df.method != "Dense"].merge(dense, on="sample_id", how="left")
    out["speedup"] = out["prefill_dense"] / out["prefill_s"]
    return out


def retention(g: pd.DataFrame, n_boot: int = 300, seed: int = 0):
    """g: rows of one (method, budget) with quality + q_dense.  Returns (ret, lo, hi)."""
    g = g.dropna(subset=["quality", "q_dense"])
    tasks = [t for t, x in g.groupby("task") if x.q_dense.mean() > 0]
    if not tasks:
        return np.nan, np.nan, np.nan
    point = np.mean([g[g.task == t].quality.mean() / g[g.task == t].q_dense.mean() for t in tasks])
    rng = np.random.default_rng(seed)
    boots = []
    per = {t: g[g.task == t][["quality", "q_dense"]].to_numpy() for t in tasks}
    for _ in range(n_boot):
        vals = []
        for t, arr in per.items():
            idx = rng.integers(0, len(arr), len(arr))
            d = arr[idx, 1].mean()
            if d > 0:
                vals.append(arr[idx, 0].mean() / d)
        if vals:
            boots.append(np.mean(vals))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def summarize(df: pd.DataFrame, keys=("method", "budget"), min_frac: float = 0.5, label: str = "") -> pd.DataFrame:
    """Cells whose sample count is below min_frac x the largest cell are dropped (e.g. left-overs of an
    earlier --preset quick run that wrote to the same files)."""
    if df.empty:
        return pd.DataFrame()
    d = attach_dense(df)
    counts = d.groupby(list(keys)).size()
    if len(counts):
        thr = min_frac * counts.max()
        small = counts[counts < thr]
        if len(small):
            print(f"[report] {label}: dropping {len(small)} under-sampled cells (n < {thr:.0f}): "
                  + ", ".join(f"{k}:{v}" for k, v in list(small.items())[:8]))
            keep = counts[counts >= thr].index
            d = d.set_index(list(keys)).loc[keep].reset_index()
    rows = []
    for k, g in d.groupby(list(keys)):
        k = k if isinstance(k, tuple) else (k,)
        ret, lo, hi = retention(g)
        longp = g[g.prompt_tokens >= LONG_PROMPT]
        rec = dict(zip(keys, k))
        rec.update(
            n=len(g), halt_frac=g.halt_frac.mean(), token_layer_reduction=g.token_layer_reduction.mean(),
            flop_reduction=g.flop_reduction.mean(), retention=ret, retention_lo=lo, retention_hi=hi,
            kl_last=g.kl_last.mean(), top1_agree=g.top1_agree.mean(),
            ans_nll_delta=g.ans_nll_delta.mean() if "ans_nll_delta" in g else np.nan,
            fidelity=g.fidelity_rougeL_vs_dense.mean() if "fidelity_rougeL_vs_dense" in g else np.nan,
            speedup_median=g.speedup.median(),
            speedup_long_median=longp.speedup.median() if len(longp) else np.nan,
            ttft_s=g.ttft_s.mean(),
        )
        for c in [c for c in g.columns if c.startswith("jaccard_")]:
            rec[c] = g[c].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def per_task(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = attach_dense(df)
    t = d.groupby(["method", "budget", "task"]).agg(quality=("quality", "mean"), q_dense=("q_dense", "mean"),
                                                     kl_last=("kl_last", "mean"), halt_frac=("halt_frac", "mean"),
                                                     speedup=("speedup", "median"), n=("sample_id", "count"))
    return t.reset_index()


def matched_quality(summary: pd.DataFrame, eff_cols=("flop_reduction", "halt_frac", "speedup_median",
                                                      "speedup_long_median")) -> pd.DataFrame:
    rows = []
    for m, g in summary.groupby("method"):
        g = g.sort_values("budget")
        pts = g[["retention"] + list(eff_cols)].to_numpy(dtype=float)
        for t in QUALITY_TARGETS:
            rec = dict(method=m, target=t)
            best = None
            for i in range(len(pts)):
                if pts[i, 0] >= t:
                    cand = pts[i, 1:]
                    if best is None or cand[0] > best[0]:
                        best = cand
                if i + 1 < len(pts) and pts[i, 0] >= t > pts[i + 1, 0]:
                    f = (pts[i, 0] - t) / (pts[i, 0] - pts[i + 1, 0])
                    cand = pts[i, 1:] + f * (pts[i + 1, 1:] - pts[i, 1:])
                    if best is None or cand[0] > best[0]:
                        best = cand
            for j, c in enumerate(eff_cols):
                rec[c] = float(best[j]) if best is not None else 0.0
            rec["reached"] = best is not None
            rec["saturated"] = bool(len(pts) and pts[-1, 0] >= t)
            rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_transport_mismatch(diag: pd.DataFrame, path: Path):
    if diag.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = ["stale", "reproject", "ridge_plain", "ridge_layer", "ridge"]
    for pol in order:
        g = diag[diag.policy == pol].sort_values("rel_layer")
        if g.empty:
            continue
        ax[0].plot(g.rel_layer, g.nmse_k, marker="o", label=pol)
        ax[1].plot(g.rel_layer, g.nmse_v, marker="o", label=pol)
    for a, t in zip(ax, ["Keys", "Values"]):
        a.set_yscale("log")
        a.set_xlabel("layers after the halting layer")
        a.set_ylabel("held-out NMSE vs. dense K/V")
        a.set_title(f"{t}: cross-layer representation mismatch")
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_frontier(summary: pd.DataFrame, oper: pd.DataFrame, path: Path):
    g = summary[summary.method == "QIRA"].sort_values("budget") if not summary.empty else pd.DataFrame()
    if g.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(g.halt_frac, g.retention, marker="o", label="QIRA (exact budget)")
    ax.fill_between(g.halt_frac, g.retention_lo, g.retention_hi, alpha=.2)
    if not oper.empty:
        for _, r in oper.iterrows():
            ax.scatter([r.halt_frac], [r.retention], marker="*", s=160, zorder=5)
            ax.annotate(str(r.method).replace("QIRA-", ""), (r.halt_frac, r.retention),
                        textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("halted tokens (fraction of prompt)")
    ax.set_ylabel("quality retained vs. Dense")
    ax2 = ax.twinx()
    ax2.plot(g.halt_frac, g.speedup_long_median, color="gray", linestyle="--", marker="s", label="prefill speed-up (long prompts)")
    ax2.set_ylabel("wall-clock prefill speed-up (x)")
    ax.set_title("QIRA operating frontier")
    ax.grid(alpha=.3)
    fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_methods(summary: pd.DataFrame, path: Path, speed_col: str = "speedup_long_median",
                speed_label: str = "wall-clock prefill speed-up, prompts >= 2K tokens (x)"):
    if summary.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in summary.groupby("method"):
        g = g.sort_values("budget")
        ls = "--" if m.endswith("+T") else "-"
        ax[0].plot(g.flop_reduction, g.retention, marker="o", linestyle=ls, label=m)
        if speed_col in g and g[speed_col].notna().any():
            ax[1].plot(g[speed_col], g.retention, marker="o", linestyle=ls, label=m)
    ax[0].set_xlabel("analytic prefill FLOP reduction")
    ax[1].set_xlabel(speed_label)
    for a in ax:
        a.set_ylabel("quality retained vs. Dense")
        a.grid(alpha=.3)
    ax[0].legend()
    ax[0].set_title("Matched token budget")
    ax[1].set_title("Quality vs. measured speed-up")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_transport_ablation(summary: pd.DataFrame, path: Path):
    if summary.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in summary.groupby("method"):
        g = g.sort_values("budget")
        ax[0].plot(g.halt_frac, g.kl_last.clip(lower=1e-6), marker="o", label=m.replace("qei+", ""))
        ax[1].plot(g.halt_frac, g.retention, marker="o", label=m.replace("qei+", ""))
    ax[0].set_yscale("log")
    ax[0].set_ylabel("KL(dense || method), next token")
    ax[1].set_ylabel("quality retained vs. Dense")
    for a in ax:
        a.set_xlabel("halted tokens (fraction of prompt)")
        a.grid(alpha=.3)
    ax[0].legend(title="KV policy (same QEI selector)")
    ax[0].set_title("Transport ablation: fidelity")
    ax[1].set_title("Transport ablation: task quality")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_scaling(sc: pd.DataFrame, path: Path):
    if sc.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    dense = sc[sc.method == "Dense"].groupby("target_len").agg(prefill=("prefill_s", "median"),
                                                               hf=("hf_prefill_s", "median")).reset_index()
    ax[0].plot(dense.target_len, dense.prefill, marker="o", label="Dense (engine)")
    if "hf" in dense and dense.hf.notna().any():
        ax[0].plot(dense.target_len, dense.hf, marker="o", linestyle=":", label="Dense (HF forward)")
    s2 = attach_dense(sc)
    for (m, b), g in s2.groupby(["method", "budget"]):
        gg = g.groupby("target_len").agg(prefill=("prefill_s", "median"), sp=("speedup", "median")).reset_index()
        ax[0].plot(gg.target_len, gg.prefill, marker="o", label=f"{m}@{b:g}")
        ax[1].plot(gg.target_len, gg.sp, marker="o", label=f"{m}@{b:g}")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("prompt length (tokens)")
        a.grid(alpha=.3)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("prefill latency (s, median)")
    ax[1].set_ylabel("prefill speed-up vs. Dense engine (x)")
    ax[1].axhline(1.0, color="k", lw=.8)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------- report


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or df.empty:
        return "_(not run)_"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def build_report(out: Path, model: str, calib_dir: Optional[Path], meta: Dict):
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    lines = [f"# QIRA report — {model}", "",
             f"Protocol `{meta.get('protocol')}` · preset `{meta.get('preset')}` · calibration source "
             f"`{meta.get('calib_source')}` · decision layer {meta.get('decision_layer')} / {meta.get('num_layers')}", "",
             "QIRA is backbone-frozen and inference-time adaptive; its K/V transport maps are calibrated "
             "offline in closed form on a pool disjoint from all evaluation data.", ""]

    # Figure 1
    diag = pd.read_csv(calib_dir / "transport_diag.csv") if calib_dir and (calib_dir / "transport_diag.csv").exists() else pd.DataFrame()
    if not diag.empty:
        fig_transport_mismatch(diag, figs / "fig1_transport_mismatch.png")
        piv = diag[diag.policy.isin(["stale", "reproject", "ridge_plain", "ridge_layer", "ridge", "mlp", "ridge(sub)"])]
        piv = piv.groupby("policy").agg(nmse_kv=("nmse_kv", "mean")).reset_index().sort_values("nmse_kv", ascending=False)
        piv.to_csv(tables / "transport_nmse.csv", index=False)
        lines += ["## 1. The problem: cross-layer K/V mismatch (calibration hold-out)", "",
                  "Mean held-out NMSE of halted-token K/V over all layers after the halting layer "
                  "(`ridge(sub)`/`mlp` are measured on the MLP subsample).", "", _md(piv, ".4f"), "",
                  "![](figures/fig1_transport_mismatch.png)", ""]
    if calib_dir and (calib_dir / "selector_alignment.csv").exists():
        al = pd.read_csv(calib_dir / "selector_alignment.csv")
        lines += ["Spearman correlation of each selector score with (a) low halting harm, harm = attention "
                  "received x ridge transport error summed over layers >= d, and (b) transportability "
                  "(−log ridge NMSE):", "", _md(al), ""]

    # clean latency (long prompts, repeats) -> preferred source of wall-clock speed-ups
    lat = load(out, "latency")
    lat_sum = pd.DataFrame()
    if not lat.empty:
        ld = attach_dense(lat)
        lat_sum = ld.groupby(["method", "budget"]).agg(
            n=("sample_id", "count"), prompt_tokens=("prompt_tokens", "mean"), halt_frac=("halt_frac", "mean"),
            flop_reduction=("flop_reduction", "mean"), prefill_s=("prefill_s", "median"),
            prefill_dense=("prefill_dense", "median"), speedup=("speedup", "median"),
            speedup_p25=("speedup", lambda x: x.quantile(.25)), speedup_p75=("speedup", lambda x: x.quantile(.75)),
            kl_last=("kl_last", "mean")).reset_index()
        if "attn_mode" in ld:
            modes = ld.groupby(["method", "budget"]).attn_mode.agg(lambda x: x.dropna().mode().iat[0] if x.notna().any() else None)
            lat_sum = lat_sum.merge(modes.rename("attn_mode").reset_index(), on=["method", "budget"], how="left")
        lat_sum.to_csv(tables / "latency.csv", index=False)

    # main
    main = load(out, "main")
    if not main.empty:
        sm = summarize(main, label="main")
        speed_col, speed_label = "speedup_long_median", "wall-clock prefill speed-up, prompts >= 2K tokens (x)"
        if not lat_sum.empty:
            sm = sm.merge(lat_sum[["method", "budget", "speedup"]].rename(columns={"speedup": "speedup_latency"}),
                          on=["method", "budget"], how="left")
            speed_col, speed_label = "speedup_latency", "wall-clock prefill speed-up, latency experiment (x)"
        sm.to_csv(tables / "main_matched_budget.csv", index=False)
        per_task(main).to_csv(tables / "main_per_task.csv", index=False)
        eff = ["flop_reduction", "halt_frac", "speedup_long_median"] + (["speedup_latency"] if "speedup_latency" in sm else [])
        mq = matched_quality(sm, eff_cols=tuple(eff))
        mq.to_csv(tables / "main_matched_quality.csv", index=False)
        fig_methods(sm, figs / "fig3_methods.png", speed_col, speed_label)
        oper = summarize(load(out, "operating"), label="operating")
        if not oper.empty:
            oper.to_csv(tables / "operating_points.csv", index=False)
        fig_frontier(sm, oper, figs / "fig2_qira_frontier.png")
        cols = ["method", "budget", "n", "halt_frac", "flop_reduction", "retention", "retention_lo", "retention_hi",
                "kl_last", "top1_agree", "fidelity", "speedup_long_median", "speedup_latency"]
        dense = main[main.method == "Dense"]
        lines += ["## 2. Matched token-reduction budget", "",
                  f"Dense: {dense.sample_id.nunique()} samples; mean Dense prefill {dense.prefill_s.mean():.3f}s; "
                  f"engine-vs-HF KL (sanity) {dense.engine_vs_hf_kl.mean() if 'engine_vs_hf_kl' in dense else float('nan'):.2e}.", "",
                  _md(sm[[c for c in cols if c in sm]].sort_values(["budget", "method"])), "",
                  "![](figures/fig3_methods.png)", "",
                  "## 3. Matched quality (accuracy-matched efficiency)", "",
                  "Largest efficiency at which each method still retains the target fraction of Dense quality "
                  "(linear interpolation along the budget sweep; `saturated` = target still met at the largest budget "
                  "tested, so the value is a lower bound).", "",
                  _md(mq), "", "## 4. QIRA operating frontier", "", "![](figures/fig2_qira_frontier.png)", "",
                  _md(oper[[c for c in cols if c in oper]] if not oper.empty else oper), ""]
        # selector x transport decomposition: same selector, native KV policy vs QIRA transport
        pairs = [("QuickSilver", "QuickSilver+T"), ("LazyLLM", "LazyLLM+T"), ("QIRA", "QIRA-L")]
        drows = []
        for a, b in pairs:
            if a in set(sm.method) and b in set(sm.method):
                for bud in sorted(set(sm.budget)):
                    ra = sm[(sm.method == a) & (sm.budget == bud)]
                    rb = sm[(sm.method == b) & (sm.budget == bud)]
                    if len(ra) and len(rb):
                        drows.append(dict(budget=bud, base=a, variant=b,
                                          retention_base=float(ra.retention.iloc[0]),
                                          retention_variant=float(rb.retention.iloc[0]),
                                          kl_base=float(ra.kl_last.iloc[0]), kl_variant=float(rb.kl_last.iloc[0])))
        if drows:
            dec = pd.DataFrame(drows)
            dec.to_csv(tables / "decomposition.csv", index=False)
            lines += ["## 4b. Decomposition: plugging QIRA transport into baseline selectors "
                      "(`+T` = same selector, ridge transport); QIRA vs QIRA-L = QEI vs learned selector", "",
                      _md(dec), ""]
    if not lat_sum.empty:
        lines += ["## 4c. Wall-clock prefill latency (LongBench prompts >= 2K tokens; warm-up + median of repeats; "
                  "no generation)", "", _md(lat_sum), ""]

    tr = load(out, "transport")
    if not tr.empty:
        st = summarize(tr, label="transport")
        st.to_csv(tables / "transport_ablation.csv", index=False)
        fig_transport_ablation(st, figs / "fig4_transport_ablation.png")
        lines += ["## 5. Transport ablation (same QEI halting set, different KV policy)", "",
                  _md(st[["method", "budget", "n", "halt_frac", "kl_last", "top1_agree", "ans_nll_delta", "retention",
                          "fidelity", "flop_reduction"]].sort_values(["budget", "method"])), "",
                  "![](figures/fig4_transport_ablation.png)", ""]

    se = load(out, "selector")
    if not se.empty:
        from qira.experiments import SELECTOR_ABLATION
        se = se[se.method.isin(["Dense"] + [f"{x}+ridge" for x in SELECTOR_ABLATION])]
        ss = summarize(se, label="selector")
        ss.to_csv(tables / "selector_ablation.csv", index=False)
        jc = [c for c in ss.columns if c.startswith("jaccard_")]
        lines += ["## 6. Selector / QEI sensitivity (same ridge transport)", "",
                  _md(ss[["method", "budget", "n", "kl_last", "ans_nll_delta", "retention", "fidelity"] + jc]
                      .sort_values(["budget", "kl_last"])), ""]

    ct = load(out, "calib_transfer")
    if not ct.empty:
        sct = summarize(ct, label="calib_transfer")
        sct.to_csv(tables / "calibration_transfer.csv", index=False)
        lines += ["## 7. Calibration transfer (transport + QEI calibrated on different pools)", "",
                  _md(sct[["method", "budget", "n", "kl_last", "ans_nll_delta", "retention", "fidelity"]]
                      .sort_values(["budget", "method"])), ""]

    sc = load(out, "scaling")
    if not sc.empty:
        fig_scaling(sc, figs / "fig5_scaling.png")
        s2 = attach_dense(sc)
        tab = s2.groupby(["target_len", "method", "budget"]).agg(
            n=("sample_id", "count"), prompt_tokens=("prompt_tokens", "mean"),
            prefill_s=("prefill_s", "median"), prefill_dense=("prefill_dense", "median"),
            speedup=("speedup", "median"), flop_reduction=("flop_reduction", "mean"),
            peak_mem_gb=("peak_mem_gb", "max"), needle_acc=("quality", "mean"),
            kl_last=("kl_last", "mean")).reset_index()
        dn = sc[sc.method == "Dense"].groupby("target_len").agg(
            dense_prefill=("prefill_s", "median"), hf_prefill=("hf_prefill_s", "median"),
            dense_mem=("peak_mem_gb", "max"), dense_needle_acc=("quality", "mean")).reset_index()
        tab.to_csv(tables / "scaling.csv", index=False)
        dn.to_csv(tables / "scaling_dense.csv", index=False)
        lines += ["## 8. Context scaling (needle retrieval; latency = median of repeats after warm-up)", "",
                  _md(dn), "", _md(tab), "", "![](figures/fig5_scaling.png)", ""]

    lines += ["## Metric definitions", "",
              "* `halt_frac`: halted tokens / prompt tokens (token reduction).",
              "* `token_layer_reduction`: fraction of (token, layer) pairs skipped.",
              "* `flop_reduction`: analytic prefill FLOPs incl. causal attention and transport overhead.",
              "* `speedup_*`: wall-clock prefill time of the Dense engine / method on the same sample "
              "(CUDA-synchronised; TTFT = prefill + first-token selection).",
              "* `kl_last`: KL(Dense || method) of the next-token distribution after the prompt.",
              "* `ans_nll_delta`: change in teacher-forced NLL of the reference answer.",
              "* `fidelity`: ROUGE-L between the method's and Dense's generations.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"[report] wrote {out / 'report.md'}")
