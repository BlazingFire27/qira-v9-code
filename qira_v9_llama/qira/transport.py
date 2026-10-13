"""
Cross-layer K/V transport for halted tokens.

A token halted at the input of layer d keeps its frozen hidden state h_d.  Every later
layer l still needs (K_l, V_l) for it.  QIRA predicts them with a closed-form linear map

    [k_l ; v_l]  ~=  x W_l ,          x = [ RMSNorm_l(h_d) , 1 ]

where k_l is the raw (pre-k_norm, pre-RoPE) key projection; RoPE (and k_norm, if the model
has one) is applied afterwards at the token's true position.  W_l is fitted by ridge
regression on dense forward passes over a calibration pool, shrinking towards the
zero-calibration "reproject" map W0_l = [W_k ; W_v ; b] (the layer's own projections):

    W_l(lambda) = (G + lambda I)^-1 (C + lambda W0_l),   G = X^T X, C = X^T T.

lambda is selected on held-out calibration prompts, independently per (layer, K/V, KV-head)
for `ridge` (QIRA default), once per layer for `ridge_layer`; `ridge_plain` shrinks towards 0
instead of W0; `mlp` adds a small residual MLP on top of `ridge`.  The held-out NMSE of every
policy, including `stale` (K/V of layer d-1 reused), is reported per layer (Figure 1).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from qira.config import QIRAConfig


class TransportStats:
    """Sufficient statistics accumulated during calibration (fp64, on device)."""

    def __init__(self, engine, cfg: QIRAConfig):
        self.e = engine
        self.cfg = cfg
        g = engine.g
        self.d = g.decision_layer
        self.layers = list(range(self.d, g.num_layers))
        D1 = g.hidden + 1
        self.kvd = g.kv_heads * g.head_dim
        C2 = 2 * self.kvd
        dev = engine.device
        f64 = torch.float64
        self.G = {s: {L: torch.zeros(D1, D1, dtype=f64, device=dev) for L in self.layers} for s in ("fit", "val")}
        self.C = {s: {L: torch.zeros(D1, C2, dtype=f64, device=dev) for L in self.layers} for s in ("fit", "val")}
        self.tt = {s: {L: torch.zeros(C2, dtype=f64, device=dev) for L in self.layers} for s in ("fit", "val")}
        self.stale_err = {L: torch.zeros(C2, dtype=f64, device=dev) for L in self.layers}
        self.count = {"fit": 0, "val": 0}
        # residual-MLP training subsample (kept on CPU)
        self.mlp_h = {"fit": [], "val": []}
        self.mlp_T = {"fit": {L: [] for L in self.layers}, "val": {L: [] for L in self.layers}}

    def raw_kv(self, L: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.e.lm.layers[L].self_attn
        return torch.cat([attn.k_proj(x), attn.v_proj(x)], dim=-1)


def _w0(engine, L: int) -> torch.Tensor:
    attn = engine.lm.layers[L].self_attn
    Wk = attn.k_proj.weight.double().T
    Wv = attn.v_proj.weight.double().T
    bk = attn.k_proj.bias.double() if attn.k_proj.bias is not None else torch.zeros(Wk.shape[1], dtype=torch.float64, device=Wk.device)
    bv = attn.v_proj.bias.double() if attn.v_proj.bias is not None else torch.zeros(Wv.shape[1], dtype=torch.float64, device=Wv.device)
    return torch.cat([torch.cat([Wk, Wv], 1), torch.cat([bk, bv])[None]], 0)


class ResidualMLP(nn.Module):
    def __init__(self, D, hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransportBank:
    VARIANTS = ("ridge_plain", "ridge_layer", "ridge", "depthweave")

    def __init__(self, engine, cfg: QIRAConfig):
        self.e = engine
        self.cfg = cfg
        self.delta: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.mlp: Dict[int, ResidualMLP] = {}
        self.meta: Dict = {}

    # ------------------------------------------------------------------ apply
    @torch.no_grad()
    def halted_kv(self, policy: str, L: int, h_frozen: torch.Tensor, cos_h, sin_h):
        """h_frozen: (1, m, D).  Returns post-RoPE k and v, each (1, KH, m, hd)."""
        e = self.e
        x = e.ln_in(L, h_frozen)[0]                      # (m, D)
        attn = e.lm.layers[L].self_attn
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        if policy == "reproject":
            return e.finish_kv(L, k, v, cos_h, sin_h)
        base = "ridge" if policy == "mlp" else policy
        if base not in self.delta:
            raise RuntimeError(f"transport '{base}' is not calibrated; run the calibrate experiment first")
        dW, db = self.delta[base][L]
        corr = x @ dW + db
        if policy == "mlp":
            m = self.mlp.get(L)
            if m is not None:
                corr = corr + m(x.float()).to(corr.dtype)
        kvd = k.shape[-1]
        return e.finish_kv(L, k + corr[:, :kvd], v + corr[:, kvd:], cos_h, sin_h)

    def gate_beta(self, policy: str) -> float:
        betas = self.meta.get("gate_beta", {})
        if policy not in betas:
            raise RuntimeError(f"{policy} has no calibrated gate (run calibrate)")
        return float(betas[policy])

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path):
        payload = {
            "delta": {var: {L: (dW.cpu(), db.cpu()) for L, (dW, db) in d.items()} for var, d in self.delta.items()},
            "mlp": {L: m.state_dict() for L, m in self.mlp.items()},
            "meta": self.meta,
        }
        torch.save(payload, path)

    def load(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        dev, dt = self.e.device, self.e.dtype
        self.delta = {var: {int(L): (dW.to(dev, dt), db.to(dev, dt)) for L, (dW, db) in d.items()}
                      for var, d in payload["delta"].items()}
        g = self.e.g
        self.mlp = {}
        for L, sd in payload.get("mlp", {}).items():
            m = ResidualMLP(g.hidden, self.cfg.mlp_hidden, 2 * g.kv_heads * g.head_dim)
            m.load_state_dict(sd)
            self.mlp[int(L)] = m.to(dev).eval()
        self.meta = payload.get("meta", {})
        return self

    # ------------------------------------------------------------------ fitting
    @torch.no_grad()
    def fit(self, stats: TransportStats) -> List[Dict]:
        e, cfg, g = self.e, self.cfg, self.e.g
        kvd, hd, KH = stats.kvd, g.head_dim, g.kv_heads
        D = g.hidden
        val_split = "val" if stats.count["val"] > 0 else "fit"
        if val_split == "fit":
            print("[transport] WARNING: no held-out calibration prompts; alpha selected on fit data")
        groups = [("K", h, slice(h * hd, (h + 1) * hd)) for h in range(KH)] + \
                 [("V", h, slice(kvd + h * hd, kvd + (h + 1) * hd)) for h in range(KH)]
        diag_rows = []
        alphas = list(cfg.ridge_alphas)
        self.delta = {v: {} for v in self.VARIANTS}
        chosen_log = {}
        for L in stats.layers:
            G, C = stats.G["fit"][L], stats.C["fit"][L]
            Gv, Cv, ttv = stats.G[val_split][L], stats.C[val_split][L], stats.tt[val_split][L]
            W0 = _w0(e, L)
            lam_base = float(torch.diagonal(G)[:D].mean().clamp_min(1e-12))
            evals, Q = torch.linalg.eigh(G)
            evals = evals.clamp_min(0.0)
            B = Q.T @ C
            B0 = Q.T @ W0

            def val_err(W):
                return (W * (Gv @ W)).sum(0) - 2.0 * (W * Cv).sum(0) + ttv

            err0 = val_err(W0)
            res_W, res_E, pl_W, pl_E = [], [], [], []
            for a in alphas:
                lam = a * lam_base
                den = (evals + lam)[:, None]
                Wr = Q @ ((B + lam * B0) / den)
                Wp = Q @ (B / den)
                res_W.append(Wr); res_E.append(val_err(Wr))
                pl_W.append(Wp); pl_E.append(val_err(Wp))
            # candidates for shrink-to-W0 variants include alpha = inf (pure reprojection)
            cand_W = res_W + [W0]
            cand_E = res_E + [err0]
            # per-head selection (QIRA default)
            W_head = torch.empty_like(W0)
            E_head = torch.empty_like(err0)
            W_plain = torch.empty_like(W0)
            E_plain = torch.empty_like(err0)
            picks = []
            for kind, h, sl in groups:
                j = int(torch.argmin(torch.stack([E[sl].sum() for E in cand_E])))
                W_head[:, sl] = cand_W[j][:, sl]
                E_head[sl] = cand_E[j][sl]
                jp = int(torch.argmin(torch.stack([E[sl].sum() for E in pl_E])))
                W_plain[:, sl] = pl_W[jp][:, sl]
                E_plain[sl] = pl_E[jp][sl]
                picks.append((alphas + [float("inf")])[j])
            jl = int(torch.argmin(torch.stack([E.sum() for E in cand_E])))
            W_layer, E_layer = cand_W[jl], cand_E[jl]
            chosen_log[L] = dict(per_head=picks, per_layer=(alphas + [float("inf")])[jl])
            for name, W in (("ridge", W_head), ("ridge_layer", W_layer), ("ridge_plain", W_plain)):
                dW = (W - W0)
                self.delta[name][L] = (dW[:D].to(e.dtype).contiguous(), dW[D].to(e.dtype).contiguous())

            def nmse(err, sl):
                return float(err[sl].sum().clamp_min(0) / ttv[sl].sum().clamp_min(1e-30))

            Ksl, Vsl = slice(0, kvd), slice(kvd, 2 * kvd)
            cands = [("reproject", err0), ("ridge_plain", E_plain), ("ridge_layer", E_layer), ("ridge", E_head)]
            if stats.count["val"] > 0:
                cands.insert(0, ("stale", stats.stale_err[L]))
            for name, err in cands:
                diag_rows.append(dict(layer=L, rel_layer=L - stats.d, policy=name,
                                      nmse_k=nmse(err, Ksl), nmse_v=nmse(err, Vsl),
                                      nmse_kv=float(err.sum().clamp_min(0) / ttv.sum().clamp_min(1e-30))))
            del G, C, Q, B, B0, res_W, pl_W

        # Construct DepthWeave (Cordoba et al., 2026): Cross-Layer Residual Factorization
        r = getattr(cfg, "depthweave_rank", 64)
        W_stacked = torch.cat([self.delta["ridge"][L][0].float() for L in stats.layers], dim=1)
        U_basis, _, _ = torch.linalg.svd(W_stacked, full_matrices=False)
        B_shared = U_basis[:, :min(r, U_basis.shape[1])]  # (D, r) shared cross-layer basis
        for L in stats.layers:
            dW_ridge, db_ridge = self.delta["ridge"][L]
            dW_proj = B_shared @ (B_shared.T @ dW_ridge.float())
            self.delta["depthweave"][L] = (dW_proj.to(e.dtype).contiguous(), db_ridge.contiguous())
            diag_rows.append(dict(layer=L, rel_layer=L - stats.d, policy="depthweave",
                                  nmse_k=float((dW_ridge.float() - dW_proj).pow(2).mean()),
                                  nmse_v=float((dW_ridge.float() - dW_proj).pow(2).mean()),
                                  nmse_kv=float((dW_ridge.float() - dW_proj).pow(2).mean())))

        self.meta["alpha_choice"] = {str(k): v for k, v in chosen_log.items()}
        self.meta["val_split"] = val_split
        return diag_rows

    def fit_mlp(self, stats: TransportStats) -> List[Dict]:
        """Residual MLP on top of `ridge` (optional nonlinear transport variant)."""
        e, cfg, g = self.e, self.cfg, self.e.g
        if not stats.mlp_h["fit"]:
            return []
        dev = e.device
        hf = torch.cat(stats.mlp_h["fit"], 0)
        hv = torch.cat(stats.mlp_h["val"], 0) if stats.mlp_h["val"] else None
        rows = []
        torch.manual_seed(0)
        for L in stats.layers:
            Tf = torch.cat(stats.mlp_T["fit"][L], 0)
            with torch.no_grad():
                xf = e.ln_in(L, hf.to(dev)[None])[0]
                base_f = self._ridge_raw(L, xf)
                Rf = (Tf.to(dev).float() - base_f.float())
                xf = xf.float()
                scale = Rf.pow(2).mean().sqrt().clamp_min(1e-6)
            mlp = ResidualMLP(g.hidden, cfg.mlp_hidden, Rf.shape[1]).to(dev)
            opt = torch.optim.AdamW(mlp.parameters(), lr=cfg.mlp_lr, weight_decay=1e-4)
            bs = min(2048, xf.shape[0])
            with torch.enable_grad():
                for step in range(cfg.mlp_steps):
                    idx = torch.randint(0, xf.shape[0], (bs,), device=dev)
                    loss = ((mlp(xf[idx]) - Rf[idx]) / scale).pow(2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            mlp.eval()
            self.mlp[L] = mlp
            if hv is not None and stats.mlp_T["val"][L]:
                with torch.no_grad():
                    Tv = torch.cat(stats.mlp_T["val"][L], 0).to(dev).float()
                    xv = e.ln_in(L, hv.to(dev)[None])[0]
                    bv = self._ridge_raw(L, xv).float()
                    mv = bv + mlp(xv.float())
                    tt = Tv.pow(2).sum()
                    rows.append(dict(layer=L, rel_layer=L - stats.d, policy="ridge(sub)",
                                     nmse_kv=float((bv - Tv).pow(2).sum() / tt)))
                    rows.append(dict(layer=L, rel_layer=L - stats.d, policy="mlp",
                                     nmse_kv=float((mv - Tv).pow(2).sum() / tt)))
        return rows

    @torch.no_grad()
    def _ridge_raw(self, L, x):
        attn = self.e.lm.layers[L].self_attn
        dW, db = self.delta["ridge"][L]
        return torch.cat([attn.k_proj(x), attn.v_proj(x)], -1) + x @ dW + db
