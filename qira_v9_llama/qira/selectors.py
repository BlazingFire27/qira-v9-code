"""
Halting selectors.  Every selector maps the decision-layer hidden states h (n, D)
to a score per token where HIGHER = MORE HALT-ABLE.  A shared `select` turns
scores into a halt set, either with an exact per-prompt budget (matched token
reduction across methods) or with a calibrated threshold (adaptive rate).

QEI (QIRA's spectral redundancy index), vectorised on GPU:
    z_p      = P^T h_p                       (fixed random projection, d = 8)
    rho_p    = cov(z_{p-w+1..p}) / tr(.)      (density-matrix analogue of the local window)
    purity   = tr(rho^2)
    rank     = 1 / purity                     (participation-ratio effective rank)
    gap      = lambda_1 - lambda_2 upper bound (one power-iteration step)  [lite]
             | (e_1 - e_2) / e_1               (batched eigvalsh)          [exact]
    stab     = cos(h_p, h_{p-1}) * exp(-||h_p - h_{p-1}|| / ||h_{p-1}||)
    QEI      = w1 purity + w2 / max(rank, 1) + w3 gap + w4 stab
"""

import math
from typing import Dict, Optional, Tuple

import torch

from qira.config import QIRAConfig
from qira.engine import HaltRequest, apply_rope, repeat_kv

QEI_VARIANT_WEIGHTS = {
    "qei": None,  # cfg.qei_weights
    "qei_equal": (0.25, 0.25, 0.25, 0.25),
    "qei_no_purity": (0.0, 0.25, 0.20, 0.20),
    "qei_no_rank": (0.35, 0.0, 0.20, 0.20),
    "qei_no_gap": (0.35, 0.25, 0.0, 0.20),
    "qei_no_stability": (0.35, 0.25, 0.20, 0.0),
    "qei_exact": None,
}

# Features of the transport-aware learned selectors (all available at the decision layer, O(n) cost).
QEI_FEATURES = ["purity", "rank_score", "gap", "stability"]
ALL_FEATURES = QEI_FEATURES + ["cosine", "neg_normdelta", "neg_drift", "neg_relnorm", "neg_lastattn", "rel_pos"]
LEARNED_SELECTORS = {"qei_fit": QEI_FEATURES, "learned": ALL_FEATURES}


class SelectorBank:
    def __init__(self, engine, cfg: QIRAConfig):
        self.e = engine
        self.cfg = cfg
        gen = torch.Generator().manual_seed(cfg.projection_seed)
        D = engine.g.hidden
        self.P = (torch.randn(D, cfg.projection_dim, generator=gen) / math.sqrt(D)).to(engine.device)
        # {"qei_fit": {"features": [...], "mu": [...], "sd": [...], "w": [...]}, "learned": {...}}
        self.learned_models: Dict[str, Dict] = {}

    @torch.no_grad()
    def features(self, h: torch.Tensor, h_prev: Optional[torch.Tensor], cos, sin) -> Dict[str, torch.Tensor]:
        comp = self.qei_components(h)
        hf = h.float()
        n = hf.shape[0]
        c, dl = self._neighbour_stats(hf)
        c = c.clone()
        c[0] = 0.0
        f = dict(comp)
        f["cosine"] = c
        f["neg_normdelta"] = -torch.nan_to_num(dl, posinf=10.0).clamp(max=10.0)
        if h_prev is not None:
            hp = h_prev.float()
            f["neg_drift"] = -((hf - hp).norm(dim=-1) / (hp.norm(dim=-1) + 1e-8))
        else:
            f["neg_drift"] = torch.zeros(n, device=h.device)
        nr = hf.norm(dim=-1)
        f["neg_relnorm"] = -(nr / (nr.mean() + 1e-8))
        f["neg_lastattn"] = -torch.log(self._attn_mass_last(h, cos, sin) + 1e-6)
        f["rel_pos"] = torch.arange(n, device=h.device, dtype=torch.float32) / max(1, n - 1)
        return f

    def _learned_score(self, name, h, h_prev, cos, sin) -> torch.Tensor:
        m = self.learned_models.get(name)
        if m is None:
            raise RuntimeError(f"selector {name} is not calibrated (run calibrate)")
        f = self.features(h, h_prev, cos, sin)
        X = torch.stack([f[k] for k in m["features"]], 1)
        mu = torch.tensor(m["mu"], device=X.device)
        sd = torch.tensor(m["sd"], device=X.device)
        w = torch.tensor(m["w"], device=X.device)
        return ((X - mu) / sd) @ w

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _neighbour_stats(hf: torch.Tensor):
        n = hf.shape[0]
        cos = torch.zeros(n, device=hf.device)
        delta = torch.full((n,), float("inf"), device=hf.device)
        if n > 1:
            cur, prev = hf[1:], hf[:-1]
            pn = prev.norm(dim=-1)
            cos[1:] = (cur * prev).sum(-1) / (cur.norm(dim=-1) * pn + 1e-8)
            delta[1:] = (cur - prev).norm(dim=-1) / (pn + 1e-8)
        return cos, delta

    def qei_components(self, h: torch.Tensor, exact: bool = False) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        hf = h.float()
        n = hf.shape[0]
        dproj = cfg.projection_dim
        w = cfg.window_w
        purity = torch.full((n,), 1.0 / dproj, device=hf.device)
        rank = torch.full((n,), float(dproj), device=hf.device)
        gap = torch.zeros(n, device=hf.device)
        if n >= w:
            z = hf @ self.P                                  # (n, d)
            win = z.unfold(0, w, 1)                          # (n-w+1, d, w): window ending at i+w-1
            win = win - win.mean(dim=-1, keepdim=True)
            cov = win @ win.transpose(1, 2) / max(w - 1, 1)  # (m, d, d)
            tr = cov.diagonal(dim1=1, dim2=2).sum(-1)
            ok = tr > 1e-12
            if exact:
                ev = torch.linalg.eigvalsh(cov + 1e-8 * torch.eye(dproj, device=hf.device))
                ev = ev.clamp_min(1e-12).flip(-1)
                p = ev / ev.sum(-1, keepdim=True)
                ent = -(p * torch.log(p + 1e-12)).sum(-1)
                r_ = torch.exp(ent)
                pu = (p ** 2).sum(-1)
                gp = (ev[:, 0] - ev[:, 1]) / (ev[:, 0] + 1e-12)
            else:
                rho = cov / tr.clamp_min(1e-12)[:, None, None]
                pu = (rho * rho).sum((1, 2))
                r_ = 1.0 / pu.clamp_min(1e-12)
                v = torch.ones(rho.shape[0], dproj, device=hf.device) / math.sqrt(dproj)
                for _ in range(max(1, cfg.power_iterations)):
                    v = torch.bmm(rho, v[:, :, None])[:, :, 0]
                    v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
                lam = (v * torch.bmm(rho, v[:, :, None])[:, :, 0]).sum(-1)
                lam2 = torch.sqrt((pu - lam ** 2).clamp_min(0.0))
                lam2 = torch.minimum(lam2, lam).clamp_min(0.0)
                gp = (lam - lam2).clamp_min(0.0)
            pu = torch.where(ok, pu, torch.full_like(pu, 1.0 / dproj))
            r_ = torch.where(ok, r_, torch.full_like(r_, float(dproj)))
            gp = torch.where(ok, gp, torch.zeros_like(gp))
            purity[w - 1:] = pu
            rank[w - 1:] = r_
            gap[w - 1:] = gp
        cos, delta = self._neighbour_stats(hf)
        stab = (cos.clamp(0, 1) * torch.exp(-delta.clamp_min(0))).clamp(0, 1)
        stab = torch.nan_to_num(stab, nan=0.0)
        if cfg.cheap_gate_enabled:
            gate = (cos >= cfg.cheap_cosine_threshold) & (delta <= cfg.cheap_norm_delta_threshold)
            gate[: max(w, 1)] = False
            purity = torch.where(gate, torch.ones_like(purity), purity)
            rank = torch.where(gate, torch.ones_like(rank), rank)
            gap = torch.where(gate, torch.ones_like(gap), gap)
        return dict(purity=purity.clamp(0, 1), rank_score=1.0 / rank.clamp_min(1.0),
                    gap=gap.clamp(0, 1), stability=stab)

    @staticmethod
    def combine(comp: Dict[str, torch.Tensor], w) -> torch.Tensor:
        return w[0] * comp["purity"] + w[1] * comp["rank_score"] + w[2] * comp["gap"] + w[3] * comp["stability"]

    def qei_weights_for(self, name: str):
        spec = QEI_VARIANT_WEIGHTS[name]
        if spec is None:
            return tuple(self.cfg.qei_weights)
        return spec

    # ---------------------------------------------------------------- scores
    @torch.no_grad()
    def score(self, name: str, h: torch.Tensor, h_prev: Optional[torch.Tensor], cos, sin) -> torch.Tensor:
        """h: (n, D) input of the decision layer; h_prev: input of the previous layer."""
        n = h.shape[0]
        if name in LEARNED_SELECTORS:
            return self._learned_score(name, h, h_prev, cos, sin).float()
        if name == "tsa":
            return torch.zeros(n, device=h.device)
        if name.startswith("qei"):
            comp = self.qei_components(h, exact=(name == "qei_exact"))
            return self.combine(comp, self.qei_weights_for(name)).float()
        hf = h.float()
        if name == "cosine":
            c, _ = self._neighbour_stats(hf)
            c[0] = -1.0
            return c
        if name == "normdelta":
            _, dl = self._neighbour_stats(hf)
            return -torch.nan_to_num(dl, posinf=1e6)
        if name == "drift":
            # QuickSilver-style: relative L2 drift of the token representation across the last layer
            if h_prev is None:
                return torch.zeros(n, device=h.device)
            hp = h_prev.float()
            return -((hf - hp).norm(dim=-1) / (hp.norm(dim=-1) + 1e-8))
        if name == "relnorm":
            nr = hf.norm(dim=-1)
            return -(nr / (nr.mean() + 1e-8))
        if name == "lastattn":
            return self._last_token_attention(h, cos, sin)
        if name == "depth_router":
            return self._depth_router_score(h, cos, sin)
        if name == "random":
            gen = torch.Generator(device="cpu").manual_seed(1234 + n)
            return torch.rand(n, generator=gen).to(h.device)
        raise ValueError(f"unknown selector {name}")

    def _depth_router_score(self, h, cos, sin) -> torch.Tensor:
        """DepthWeave (Cordoba et al., 2026): token-adaptive depth router based on multi-head attention sensitivity probe."""
        e = self.e
        d = e.g.decision_layer
        attn = e.lm.layers[d].self_attn
        x = e.ln_in(d, h[None])
        n = x.shape[1]
        step = max(1, n // 32)
        q_idx = torch.arange(0, n, step, device=h.device)
        q = attn.q_proj(x[:, q_idx]).view(1, len(q_idx), e.g.heads, e.g.head_dim)
        k = attn.k_proj(x).view(1, n, e.g.kv_heads, e.g.head_dim)
        if e.lm.has_qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q = apply_rope(q.transpose(1, 2), cos[q_idx], sin[q_idx])
        k = apply_rope(k.transpose(1, 2), cos, sin)
        k = repeat_kv(k, e.n_rep)
        s = (q.float() @ k.float().transpose(-1, -2)) * e.lm.scaling
        causal = torch.arange(n, device=h.device)[None, :] <= q_idx[:, None]
        s = s.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs = torch.softmax(s, dim=-1)
        # Average attention mass across queries and heads received by each token
        sensitivity = probs[0].mean(dim=(0, 1))
        return -sensitivity

    def _last_token_attention(self, h, cos, sin) -> torch.Tensor:
        """LazyLLM: attention mass the last token assigns to each position at the decision layer."""
        return -self._attn_mass_last(h, cos, sin)

    def _attn_mass_last(self, h, cos, sin) -> torch.Tensor:
        e = self.e
        d = e.g.decision_layer
        attn = e.lm.layers[d].self_attn
        x = e.ln_in(d, h[None])
        n = x.shape[1]
        q = attn.q_proj(x[:, -1:]).view(1, 1, e.g.heads, e.g.head_dim)
        k = attn.k_proj(x).view(1, n, e.g.kv_heads, e.g.head_dim)
        if e.lm.has_qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q = apply_rope(q.transpose(1, 2), cos[-1:], sin[-1:])
        k = apply_rope(k.transpose(1, 2), cos, sin)
        k = repeat_kv(k, e.n_rep)
        s = (q.float() @ k.float().transpose(-1, -2)) * e.lm.scaling
        return torch.softmax(s, dim=-1)[0, :, 0, :].mean(0)

    # ---------------------------------------------------------------- halting set
    def select(self, scores: torch.Tensor, n: int, lo: int, hi: int, req: HaltRequest) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
        if hi <= lo:
            return mask
        s = scores[lo:hi]
        if req.mode == "budget":
            k = int(round(float(req.value) * (hi - lo)))
            k = max(0, min(k, hi - lo))
            if k > 0:
                idx = torch.topk(torch.nan_to_num(s, nan=-1e9), k, sorted=False).indices + lo
                mask[idx] = True
            return mask
        if req.mode == "threshold":
            if req.tau is None:
                raise ValueError("threshold mode needs a calibrated tau")
            halt = s >= req.tau
            if req.tau_low is not None and req.tau_low < req.tau:
                # hysteresis: once halted, keep halting while score stays above tau_low
                h_cpu = halt.cpu().tolist()
                s_cpu = s.float().cpu().tolist()
                prev = False
                for i in range(len(h_cpu)):
                    cur = (s_cpu[i] >= req.tau_low) if prev else h_cpu[i]
                    h_cpu[i] = cur
                    prev = cur
                halt = torch.tensor(h_cpu, dtype=torch.bool, device=scores.device)
            mask[lo:hi] = halt
            return mask
        raise ValueError(f"unknown halting mode {req.mode}")
