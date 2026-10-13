"""
Offline, closed-form calibration (backbone stays frozen).

One dense pass over a calibration pool (disjoint from all evaluation data) produces:
  1. transport sufficient statistics  -> ridge / ridge_layer / ridge_plain (+ residual MLP) maps;
  2. per-layer held-out transport error of every KV policy (Figure 1: the mismatch problem);
  3. selector score distributions      -> threshold-mode quantiles tau(r) for target halt rate r;
  4. QEI components vs. token "transportability" (how well the frozen state predicts its future
     K/V)                               -> learned non-negative QEI weights + selector alignment.
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from qira.config import QIRAConfig, SELECTORS
from qira.selectors import LEARNED_SELECTORS, QEI_VARIANT_WEIGHTS, SelectorBank
from qira.transport import TransportBank, TransportStats

RATE_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


GATE_BETAS = [0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
GATE_BUDGETS = [0.5, 0.8]


@torch.no_grad()
def calibrate_gate(engine, formatter, pool, cfg, bank, max_tokens, max_prompts: int = 8) -> List[Dict]:
    """Choose the attention-logit bias of `ridge_gated` on held-out calibration prompts (min mean KL).
    beta = 0 is pure transport; beta -> -inf recovers `drop`."""
    from qira.engine import HaltRequest
    prev = engine.transport
    engine.transport = bank
    held = [ex for i, ex in enumerate(pool) if (i % cfg.val_every) == (cfg.val_every - 1)]
    extra = [ex for i, ex in enumerate(pool) if (i % cfg.val_every) != (cfg.val_every - 1)]
    prompts = (held + extra)[:max_prompts]
    rows = []
    kl = {b: [] for b in GATE_BETAS + ["drop"]}
    for ex in prompts:
        ids_l = formatter.encode(ex, max_context=max_tokens)
        lo, hi = engine.eligible_range(len(ids_l))
        if hi - lo < 8:
            continue
        ids = torch.tensor([ids_l], device=engine.device)
        ref = torch.log_softmax(engine.prefill(ids, None).last_logits[0], -1)
        for bud in GATE_BUDGETS:
            for b in GATE_BETAS + ["drop"]:
                req = (HaltRequest("qei", "drop", "budget", bud) if b == "drop" else
                       HaltRequest("qei", "ridge_gated", "budget", bud, gate_beta=float(b)))
                lp = torch.log_softmax(engine.prefill(ids, req).last_logits[0], -1)
                kl[b].append(float((ref.exp() * (ref - lp)).sum()))
    for b, v in kl.items():
        rows.append(dict(beta=b, mean_kl=float(np.mean(v)) if v else float("nan"), n=len(v)))
    cand = [r for r in rows if r["beta"] != "drop" and np.isfinite(r["mean_kl"])]
    best = min(cand, key=lambda r: r["mean_kl"])["beta"] if cand else -1.0
    bank.meta["gate_beta"] = {"ridge_gated": float(best)}
    engine.transport = prev
    print(f"[calibration] ridge_gated beta = {best}  (KL by beta: "
          + ", ".join(f"{r['beta']}:{r['mean_kl']:.4f}" for r in rows) + ")")
    return rows


@torch.no_grad()
def run_calibration(engine, selectors: SelectorBank, formatter, pool: List[Dict], cfg: QIRAConfig,
                    out_dir: Path, max_tokens: int, fit_mlp: bool = True) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = engine.g
    d = g.decision_layer
    stats = TransportStats(engine, cfg)
    rng = np.random.default_rng(0)
    per_prompt_mlp = int(math.ceil(cfg.mlp_max_tokens / max(1, len(pool))))
    encoded = []
    n_tok = 0
    # ---------------- pass 1: transport sufficient statistics ----------------
    for i, ex in enumerate(pool):
        ids_list = formatter.encode(ex, max_context=max_tokens)
        n = len(ids_list)
        lo, hi = engine.eligible_range(n)
        if hi - lo < 8 or n < cfg.window_w + 2:
            continue
        split = "val" if (i % cfg.val_every) == (cfg.val_every - 1) else "fit"
        encoded.append((ids_list, split))
        ids = torch.tensor([ids_list], device=engine.device)
        st: Dict = {}
        m = hi - lo
        sub = None
        if fit_mlp:
            k = min(m, per_prompt_mlp)
            sub = torch.tensor(np.sort(rng.choice(m, size=k, replace=False)), device=engine.device)

        def capture(L, h):
            if L == d - 1:
                st["h_dm1"] = h
            if L == d:
                st["h_d"] = h
                x = engine.ln_in(d - 1, st["h_dm1"][:, lo:hi])[0]
                st["T_stale"] = stats.raw_kv(d - 1, x).float()
                if sub is not None:
                    stats.mlp_h[split].append(h[0, lo:hi][sub].detach().to("cpu"))
            if L >= d:
                X = engine.ln_in(L, st["h_d"][:, lo:hi])[0]
                T = stats.raw_kv(L, engine.ln_in(L, h[:, lo:hi])[0]).float()
                Xf = torch.cat([X.float(), torch.ones(m, 1, device=X.device)], 1)
                stats.G[split][L] += (Xf.T @ Xf).double()
                stats.C[split][L] += (Xf.T @ T).double()
                stats.tt[split][L] += (T * T).sum(0).double()
                if split == "val":
                    stats.stale_err[L] += ((T - st["T_stale"]) ** 2).sum(0).double()
                if sub is not None:
                    stats.mlp_T[split][L].append(T[sub].to(engine.dtype).to("cpu"))

        engine.prefill(ids, None, capture=capture)
        stats.count[split] += 1
        n_tok += m
        if (len(encoded) % 8) == 0:
            print(f"[calibration] pass 1: {len(encoded)}/{len(pool)} prompts, {n_tok} eligible tokens")
    used = len(encoded)
    if used == 0:
        raise RuntimeError("calibration pool produced no usable prompts")
    print(f"[calibration] collected {n_tok} eligible tokens from {used} prompts "
          f"(fit={stats.count['fit']}, held-out={stats.count['val']})")

    bank = TransportBank(engine, cfg)
    diag = bank.fit(stats)
    if fit_mlp:
        diag += bank.fit_mlp(stats)
    bank.meta.update(dict(n_prompts=used, n_tokens=n_tok, decision_layer=d))
    del stats
    if engine.device.type == "cuda":
        torch.cuda.empty_cache()
    gate_rows = calibrate_gate(engine, formatter, pool, cfg, bank, max_tokens)
    bank.save(out_dir / "transport.pt")
    pd.DataFrame(diag).to_csv(out_dir / "transport_diag.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out_dir / "gate_selection.csv", index=False)

    # ---------------- pass 2: selector features, scores and per-token halting harm ----------------
    feats, scores, harm, transp, splits = harm_pass(engine, selectors, bank, encoded)
    y = -np.log(harm + 1e-6)
    learned = fit_learned_selectors(feats, y, splits)
    selectors.learned_models = learned
    # learned selector scores for thresholds / alignment
    for name, mdl in learned.items():
        X = np.stack([feats[k] for k in mdl["features"]], 1)
        scores[name] = ((X - np.array(mdl["mu"])) / np.array(mdl["sd"])) @ np.array(mdl["w"])

    thresholds = {}
    for name, sc in scores.items():
        sc = sc[np.isfinite(sc)]
        tab = {}
        for r in RATE_GRID:
            tau = float(np.quantile(sc, 1.0 - r))
            low = float(np.quantile(sc, max(0.0, 1.0 - r - cfg.hysteresis_band))) if name.startswith("qei") else None
            tab[f"{r:.2f}"] = dict(tau=tau, tau_low=low)
        thresholds[name] = tab

    align = []
    for name, sc in scores.items():
        align.append(dict(selector=name, spearman_vs_low_harm=_spearman(sc, y),
                          spearman_vs_transportability=_spearman(sc, transp)))
    for k, v in feats.items():
        align.append(dict(selector=f"feature:{k}", spearman_vs_low_harm=_spearman(v, y),
                          spearman_vs_transportability=_spearman(v, transp)))
    pd.DataFrame(align).to_csv(out_dir / "selector_alignment.csv", index=False)

    record = dict(
        n_prompts=used, n_eligible_tokens=int(n_tok), decision_layer=d,
        learned_selectors=learned, thresholds=thresholds, alpha_choice=bank.meta.get("alpha_choice"),
        gate_beta=bank.meta.get("gate_beta"),
        semantics=("threshold tau(r) = (1-r)-quantile of the selector score over eligible calibration "
                   "tokens, so r is the expected halt rate for every selector; learned selectors predict "
                   "-log(halting harm), harm = sum over layers >= d of (attention mass received) x "
                   "(ridge transport error) of the token"),
    )
    (out_dir / "calibration.json").write_text(json.dumps(record, indent=1))
    for name, mdl in learned.items():
        print(f"[calibration] {name}: held-out Spearman {mdl['heldout_spearman']:.3f}; weights " +
              ", ".join(f"{f}={w:+.2f}" for f, w in zip(mdl["features"], mdl["w"])))
    return dict(bank=bank, record=record)


@torch.no_grad()
def harm_pass(engine, selectors: SelectorBank, bank: TransportBank, encoded, max_prompts: int = 48):
    """Per eligible token: decision-layer features, all selector scores, and halting harm
    sum_{l>=d} mass_l(t) * err_l(t)  (attention received x ridge transport error)."""
    g = engine.g
    d = g.decision_layer
    from qira.experiments import SELECTOR_ABLATION
    names = [s for s in SELECTOR_ABLATION if s not in LEARNED_SELECTORS]
    feats: Dict[str, List[np.ndarray]] = {}
    scores: Dict[str, List[np.ndarray]] = {s: [] for s in names}
    harm_l, transp_l, split_l = [], [], []
    for ids_list, split in encoded[:max_prompts]:
        n = len(ids_list)
        lo, hi = engine.eligible_range(n)
        m = hi - lo
        ids = torch.tensor([ids_list], device=engine.device)
        cos, sin = engine.rope(n)
        st: Dict = {}
        harm = torch.zeros(m, device=engine.device)
        terr = torch.zeros(m, device=engine.device)

        def capture(L, h):
            if L == d - 1:
                st["h_dm1"] = h
            if L == d:
                st["h_d"] = h
            if L >= d:
                x = engine.ln_in(L, h)
                q, k, _ = engine.qkv(L, x, cos, sin)
                kr = k.repeat_interleave(engine.n_rep, dim=1).float()
                mass = torch.zeros(n, device=engine.device)
                kpos = torch.arange(n, device=engine.device)
                for s0 in range(0, n, 512):
                    e0 = min(n, s0 + 512)
                    sc = (q[:, :, s0:e0].float() @ kr.transpose(-1, -2)) * engine.lm.scaling
                    qpos = torch.arange(s0, e0, device=engine.device)
                    sc = sc.masked_fill((kpos[None, :] > qpos[:, None])[None, None], float("-inf"))
                    mass += torch.softmax(sc, -1).sum((0, 1, 2)) / g.heads
                X = engine.ln_in(L, st["h_d"][:, lo:hi])[0]
                P = bank._ridge_raw(L, X).float()
                T = torch.cat([engine.lm.layers[L].self_attn.k_proj(x[0, lo:hi]),
                               engine.lm.layers[L].self_attn.v_proj(x[0, lo:hi])], -1).float()
                err = ((P - T) ** 2).sum(-1) / (T * T).sum(-1).clamp_min(1e-12)
                harm.add_(mass[lo:hi] * err)
                terr.add_(err)

        engine.prefill(ids, None, capture=capture)
        h_d, h_dm1 = st["h_d"][0], st["h_dm1"][0]
        f = selectors.features(h_d, h_dm1, cos, sin)
        for k, v in f.items():
            feats.setdefault(k, []).append(v[lo:hi].float().cpu().numpy())
        for s in names:
            scores[s].append(selectors.score(s, h_d, h_dm1, cos, sin)[lo:hi].float().cpu().numpy())
        harm_l.append(harm.cpu().numpy())
        transp_l.append(-np.log((terr / (g.num_layers - d)).clamp_min(1e-8).cpu().numpy()))
        split_l.append(np.full(m, split == "val"))
    cat = lambda dct: {k: np.concatenate(v) for k, v in dct.items()}
    print(f"[calibration] pass 2 (harm/features): {len(split_l)} prompts")
    return cat(feats), cat(scores), np.concatenate(harm_l), np.concatenate(transp_l), np.concatenate(split_l)


def fit_learned_selectors(feats: Dict[str, np.ndarray], y: np.ndarray, is_val: np.ndarray) -> Dict:
    """Closed-form ridge regression of -log(harm) on standardised decision-layer features."""
    out = {}
    fit = ~is_val if is_val.any() and (~is_val).any() else np.ones_like(is_val, dtype=bool)
    val = is_val if is_val.any() and (~is_val).any() else np.ones_like(is_val, dtype=bool)
    for name, cols in LEARNED_SELECTORS.items():
        X = np.stack([feats[c] for c in cols], 1).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mu = X[fit].mean(0)
        sd = X[fit].std(0) + 1e-6
        Z = (X - mu) / sd
        yc = y[fit] - y[fit].mean()
        lam = 1e-2 * fit.sum()
        w = np.linalg.solve(Z[fit].T @ Z[fit] + lam * np.eye(Z.shape[1]), Z[fit].T @ yc)
        out[name] = dict(features=list(cols), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(),
                         heldout_spearman=_spearman(Z[val] @ w, y[val]))
    return out


def load_calibration(engine, selectors: SelectorBank, cfg: QIRAConfig, out_dir: Path) -> Dict:
    record = json.loads((out_dir / "calibration.json").read_text())
    bank = TransportBank(engine, cfg).load(out_dir / "transport.pt")
    selectors.learned_models = record.get("learned_selectors", {})
    return dict(bank=bank, record=record)
