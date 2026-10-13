"""
QIRA inference engine.

Prefill with intermediate-layer token halting:

    layers [0, d)      : dense, all tokens (flash / SDPA causal)
    input of layer d   : selector scores tokens  ->  halt set H (exact budget or calibrated threshold)
    layers [d, L)      : only active tokens A are processed (Q/K/V/O + MLP);
                         halted tokens still need K/V at every later layer, produced by a KV policy:
                           stale       K/V of layer d-1 reused unchanged (classic "KV skipping")
                           drop        halted tokens removed from attention at layers >= d (pruning)
                           reproject   layer-l k/v projections applied to the frozen hidden state
                           ridge*/mlp  QIRA transport: calibrated cross-layer map  h_d -> (K_l, V_l)
Decode uses the engine's own pre-allocated KV cache (no dependence on HF Cache APIs),
so every KV policy, including `drop`, is honoured consistently during generation.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from qira.config import QIRAConfig
from qira.token_sparse import TokenSparseAttentionWrapper


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (1, heads, m, hd); cos/sin: (m, hd)."""
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class KVCache:
    """Pre-allocated per-layer K (post-RoPE) / V buffers with optional per-layer key validity."""

    def __init__(self, num_layers, kv_heads, head_dim, capacity, device, dtype):
        self.k = [torch.empty(1, kv_heads, capacity, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.empty(1, kv_heads, capacity, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
        self.valid: List[Optional[torch.Tensor]] = [None] * num_layers
        self.length = 0
        self.capacity = capacity

    def any_invalid(self) -> bool:
        return any(v is not None for v in self.valid)


@dataclass
class HaltRequest:
    selector: str
    transport: str
    mode: str = "budget"              # "budget" (exact fraction of eligible tokens) or "threshold"
    value: float = 0.5                # budget fraction, or target halt rate r for threshold mode
    tau: Optional[float] = None       # threshold mode: calibrated score threshold
    tau_low: Optional[float] = None   # threshold mode: hysteresis exit threshold
    gate_beta: Optional[float] = None # override of the calibrated logit bias for *_gated policies
    seed: int = 0


@dataclass
class PrefillResult:
    cache: KVCache
    last_logits: torch.Tensor
    n: int
    halt_mask: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    eligible: int = 0
    timings: Dict[str, float] = field(default_factory=dict)
    attn_pairs_after: int = 0         # number of (query, key) pairs actually attended at each layer >= d
    decision_state: Any = None


class Engine:
    def __init__(self, lm, cfg: QIRAConfig):
        self.lm = lm
        self.cfg = cfg
        self.g = lm.geom
        self.device = lm.device
        self.dtype = lm.dtype
        self.n_rep = self.g.heads // self.g.kv_heads
        self._cos = None
        self._sin = None
        self.selectors = None      # set by run_benchmark (qira.selectors.SelectorBank)
        self.transport = None      # set by run_benchmark (qira.transport.TransportBank)
        self.oracle_cache = None   # dense KVCache of the same prompt (only for the "oracle" policy)
        self.attn_backend = "mask"  # "flex" (block-sparse FlexAttention) or "mask" (chunked masked SDPA)
        self._flex = None
        self._create_block_mask = None
        self.flex_runtime_failure = False
        self.tsa = TokenSparseAttentionWrapper(
            num_layers=self.g.num_layers,
            decision_layer_idx=self.g.decision_layer,
            coverage=0.01,
            window_size=128,
            kernel_size=7,
            min_tokens=512,
        )

    # ------------------------------------------------------------------ sparse-attention backend
    def enable_flex(self) -> bool:
        """Compile FlexAttention and validate it against masked SDPA; fall back to "mask" on any problem."""
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            import torch._dynamo
            # every prompt has a different (n_active, n) shape: allow many dynamic-shape cache entries so dynamo
            # never silently falls back to the slow eager reference implementation of flex_attention
            for attr, val in (("cache_size_limit", 512), ("accumulated_cache_size_limit", 4096)):
                if hasattr(torch._dynamo.config, attr):
                    setattr(torch._dynamo.config, attr, max(getattr(torch._dynamo.config, attr), val))
            self._flex = torch.compile(flex_attention, dynamic=True)
            self._create_block_mask = create_block_mask
            g = self.g
            torch.manual_seed(0)
            for n, n_a in ((700, 333), (2600, 900)):
                keep = torch.sort(torch.randperm(n, device=self.device)[:n_a]).values
                keep[-1] = n - 1
                q = torch.randn(1, g.heads, n_a, g.head_dim, device=self.device, dtype=self.dtype)
                Kb = torch.randn(1, g.kv_heads, n + 37, g.head_dim, device=self.device, dtype=self.dtype)
                Vb = torch.randn_like(Kb)
                out = self._flex_attend(q, Kb[:, :, :n], Vb[:, :, :n], self._flex_block_mask(keep, n))
                mask = (torch.arange(n, device=self.device)[None, :] <= keep[:, None])[None, None]
                ref = self.attend(q, Kb[:, :, :n], Vb[:, :, :n], mask=mask)
                err = float((out.float() - ref.float()).abs().max())
                tol = 3e-2 if self.dtype in (torch.bfloat16, torch.float16) else 1e-4
                if not (err < tol):
                    raise RuntimeError(f"flex mismatch {err:.3e}")
            self.attn_backend = "flex"
            print("[engine] sparse attention backend: FlexAttention (validated)")
            return True
        except Exception as ex:
            self.attn_backend = "mask"
            print(f"[engine] FlexAttention unavailable ({type(ex).__name__}: {str(ex)[:160]}); "
                  "using chunked masked SDPA")
            return False

    def _flex_block_mask(self, qpos: torch.Tensor, kv_len: int):
        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= qpos[q_idx]
        try:
            return self._create_block_mask(mask_mod, None, None, qpos.shape[0], kv_len, device=qpos.device,
                                           _compile=True)
        except TypeError:
            return self._create_block_mask(mask_mod, None, None, qpos.shape[0], kv_len, device=qpos.device)

    def _flex_attend(self, q, K, V, block_mask):
        return self._flex(q, K, V, block_mask=block_mask, scale=self.lm.scaling, enable_gqa=self.n_rep > 1)

    # ------------------------------------------------------------------ RoPE
    def rope(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or self._cos.shape[0] < n:
            cap = max(n, 4096)
            if self._cos is not None:
                cap = max(cap, 2 * self._cos.shape[0])
            pos = torch.arange(cap, device=self.device).unsqueeze(0)
            dummy = torch.zeros(1, 1, self.g.head_dim, device=self.device, dtype=self.dtype)
            cos, sin = self.lm.rotary_emb(dummy, pos)
            self._cos, self._sin = cos[0].to(self.dtype), sin[0].to(self.dtype)
        return self._cos[:n], self._sin[:n]

    # ------------------------------------------------------------------ layer pieces
    def ln_in(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        return self.lm.layers[layer_idx].input_layernorm(h)

    def kv_from_normed(self, layer_idx: int, x: torch.Tensor, cos, sin):
        """x: (1, m, D) already input-normed. Returns post-RoPE k and v: (1, KH, m, hd)."""
        attn = self.lm.layers[layer_idx].self_attn
        m = x.shape[1]
        k = attn.k_proj(x).view(1, m, self.g.kv_heads, self.g.head_dim)
        v = attn.v_proj(x).view(1, m, self.g.kv_heads, self.g.head_dim)
        if self.lm.has_qk_norm:
            k = attn.k_norm(k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return apply_rope(k, cos, sin), v

    def finish_kv(self, layer_idx: int, k_pre: torch.Tensor, v: torch.Tensor, cos, sin):
        """k_pre/v: (m, KH*hd) raw projections (pre k_norm, pre RoPE). Returns post-RoPE k, v (1,KH,m,hd)."""
        m = k_pre.shape[0]
        k = k_pre.view(1, m, self.g.kv_heads, self.g.head_dim)
        v = v.view(1, m, self.g.kv_heads, self.g.head_dim)
        if self.lm.has_qk_norm:
            k = self.lm.layers[layer_idx].self_attn.k_norm(k)
        return apply_rope(k.transpose(1, 2), cos, sin), v.transpose(1, 2)

    def qkv(self, layer_idx: int, x: torch.Tensor, cos, sin):
        attn = self.lm.layers[layer_idx].self_attn
        m = x.shape[1]
        q = attn.q_proj(x).view(1, m, self.g.heads, self.g.head_dim)
        k = attn.k_proj(x).view(1, m, self.g.kv_heads, self.g.head_dim)
        v = attn.v_proj(x).view(1, m, self.g.kv_heads, self.g.head_dim)
        if self.lm.has_qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q, K, V, mask=None, is_causal=False):
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        return F.scaled_dot_product_attention(q, K, V, attn_mask=mask, is_causal=is_causal,
                                              scale=self.lm.scaling)

    def finish_layer(self, layer_idx: int, residual: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        layer = self.lm.layers[layer_idx]
        m = attn_out.shape[2]
        o = attn_out.transpose(1, 2).reshape(1, m, self.g.heads * self.g.head_dim)
        h = residual + layer.self_attn.o_proj(o)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def dense_layer(self, layer_idx: int, h: torch.Tensor, cos, sin):
        x = self.ln_in(layer_idx, h)
        q, k, v = self.qkv(layer_idx, x, cos, sin)
        out = self.attend(q, k, v, is_causal=True)
        return self.finish_layer(layer_idx, h, out), k, v

    def logits_last(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.lm.lm_head(self.lm.final_norm(h_last)).float()

    # ------------------------------------------------------------------ eligibility
    def eligible_range(self, n: int) -> Tuple[int, int]:
        lo = min(self.cfg.protect_prefix, n)
        hi = max(lo, n - max(1, self.cfg.protect_suffix))
        return lo, hi

    def _check_len(self, n: int):
        if self.lm.sliding_window is not None and n > self.lm.sliding_window:
            raise RuntimeError(f"prompt length {n} exceeds sliding window {self.lm.sliding_window}; "
                               "sliding-window attention is not implemented in the QIRA engine")

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, halt: Optional[HaltRequest] = None, extra_capacity: int = 0,
                capture: Optional[Callable] = None, keep_decision: bool = False) -> PrefillResult:
        try:
            return self._prefill(ids, halt, extra_capacity, capture, keep_decision)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as ex:
            if self.attn_backend != "flex" or halt is None:
                raise
            print(f"[engine] FlexAttention failed at runtime ({type(ex).__name__}: {str(ex)[:120]}); "
                  "switching to chunked masked SDPA for the rest of the run")
            self.attn_backend = "mask"
            self.flex_runtime_failure = True
            return self._prefill(ids, halt, extra_capacity, capture, keep_decision)

    def _prefill(self, ids: torch.Tensor, halt: Optional[HaltRequest] = None, extra_capacity: int = 0,
                 capture: Optional[Callable] = None, keep_decision: bool = False) -> PrefillResult:
        """
        ids: (1, n) on device.  halt=None -> dense.  capture(layer_idx, h_in) is called with the input
        hidden state of every layer in dense mode (used by calibration).  keep_decision stores the
        decision-layer states on the result (selector-overlap analysis).
        """
        g, cfg = self.g, self.cfg
        n = ids.shape[1]
        self._check_len(n)
        cos, sin = self.rope(n)
        cache = KVCache(g.num_layers, g.kv_heads, g.head_dim, n + extra_capacity + 1, self.device, self.dtype)
        h = self.lm.embed_tokens(ids)
        d = g.decision_layer
        timings: Dict[str, Any] = {}
        h_prev = None
        for L in range(d):
            if capture is not None:
                capture(L, h)
            h_prev = h
            h, k, v = self.dense_layer(L, h, cos, sin)
            cache.k[L][:, :, :n] = k
            cache.v[L][:, :, :n] = v

        lo, hi = self.eligible_range(n)
        res = PrefillResult(cache=cache, last_logits=None, n=n, eligible=hi - lo, timings=timings)
        halt_mask = None
        if halt is not None:
            if halt.selector == "tsa" or halt.transport == "sparse_attn":
                t0 = time.perf_counter()
                res.timings["attn_mode"] = "token_sparse"
                cov = max(0.001, min(0.99, float(halt.value)))
                self.tsa.coverage = cov
                total_retained_pairs = 0
                for L in range(d, g.num_layers):
                    x = self.ln_in(L, h)
                    q, k, v = self.qkv(L, x, cos, sin)
                    k_rep = repeat_kv(k, self.n_rep)
                    v_rep = repeat_kv(v, self.n_rep)
                    indices = self.tsa.get_sparse_indices(q, k_rep)
                    if indices is None:
                        out = self.attend(q, k, v, is_causal=True)
                        total_retained_pairs += n * (n + 1) // 2
                    else:
                        c_len = indices.shape[-1]
                        total_retained_pairs += int(indices.shape[1] * c_len * (c_len + 1) // (2 * g.heads))
                        ind_exp = indices.unsqueeze(-1).expand(-1, -1, -1, g.head_dim)
                        q_c = torch.gather(q, dim=2, index=ind_exp)
                        k_c = torch.gather(k_rep, dim=2, index=ind_exp)
                        v_c = torch.gather(v_rep, dim=2, index=ind_exp)
                        c_mask = (torch.arange(c_len, device=self.device)[None, :] <= torch.arange(c_len, device=self.device)[:, None])[None, None]
                        scores_c = (q_c.float() @ k_c.float().transpose(-1, -2)) * self.lm.scaling
                        scores_c = scores_c.masked_fill(~c_mask, float("-inf"))
                        probs_c = torch.softmax(scores_c, dim=-1).to(self.dtype)
                        out_c = probs_c @ v_c
                        out_full = torch.zeros((1, g.heads, n, g.head_dim), device=self.device, dtype=self.dtype)
                        out_full.scatter_(dim=2, index=ind_exp, src=out_c)
                        out = out_full
                    cache.k[L][:, :, :n] = k
                    cache.v[L][:, :, :n] = v
                    h = self.finish_layer(L, h, out)
                cache.length = n
                res.last_logits = self.logits_last(h[:, -1, :])
                res.attn_pairs_after = total_retained_pairs // max(1, g.num_layers - d)
                res.timings["select_s"] = time.perf_counter() - t0
                return res

            t0 = time.perf_counter()
            scores = self.selectors.score(halt.selector, h[0], h_prev[0] if h_prev is not None else None, cos, sin)
            halt_mask = self.selectors.select(scores, n, lo, hi, halt)
            res.scores = scores
            res.halt_mask = halt_mask
            if keep_decision:
                res.decision_state = (h[0], h_prev[0] if h_prev is not None else None, cos, sin)
            timings["select_s"] = time.perf_counter() - t0
            if int(halt_mask.sum()) == 0:
                halt_mask = None

        if halt_mask is None:
            for L in range(d, g.num_layers):
                if capture is not None:
                    capture(L, h)
                h, k, v = self.dense_layer(L, h, cos, sin)
                cache.k[L][:, :, :n] = k
                cache.v[L][:, :, :n] = v
            cache.length = n
            res.last_logits = self.logits_last(h[:, -1, :])
            res.attn_pairs_after = n * (n + 1) // 2
            res.timings = timings
            return res

        # ---------------- sparse layers ----------------
        keep_idx = (~halt_mask).nonzero(as_tuple=True)[0]
        halt_idx = halt_mask.nonzero(as_tuple=True)[0]
        h_frozen = h[:, halt_idx, :]                     # (1, m_h, D) frozen hidden state of halted tokens
        h_act = h[:, keep_idx, :]
        cos_a, sin_a = cos[keep_idx], sin[keep_idx]
        cos_h, sin_h = cos[halt_idx], sin[halt_idx]
        policy = halt.transport
        gate_beta = None
        if policy.endswith("_gated"):
            gate_beta = self.transport.gate_beta(policy) if halt.gate_beta is None else halt.gate_beta
            policy = policy[: -len("_gated")]
        drop = policy == "drop"
        key_valid = (~halt_mask) if drop else None
        # stale source: layer d-1 K/V of the halted tokens (post-RoPE, same positions)
        stale_k = cache.k[d - 1][:, :, halt_idx].clone() if policy == "stale" else None
        stale_v = cache.v[d - 1][:, :, halt_idx].clone() if policy == "stale" else None
        # attention pair count for FLOP accounting
        if drop:
            kept_before = torch.cumsum((~halt_mask).long(), 0)
            res.attn_pairs_after = int(kept_before[keep_idx].sum())
        else:
            res.attn_pairs_after = int((keep_idx + 1).sum())

        chunk = max(1, int(cfg.attn_query_chunk))
        n_a = keep_idx.shape[0]
        key_pos = torch.arange(n, device=self.device)
        # attention mode for layers >= d:
        #   drop          -> keys are exactly the active tokens: plain causal flash attention on the compact sequence
        #   flex          -> one block-sparse FlexAttention kernel (causal w.r.t. original positions)
        #   mask          -> chunked SDPA with an explicit mask (also used for *_gated)
        if drop:
            mode = "compact"
        elif self.attn_backend == "flex" and gate_beta is None:
            mode = "flex"
            block_mask = self._flex_block_mask(keep_idx, n)
        else:
            mode = "mask"
        res.timings["attn_mode"] = mode
        masks = []
        for s in (range(0, n_a, chunk) if mode == "mask" else []):
            qi = keep_idx[s:s + chunk]
            kmax = int(qi[-1]) + 1
            mk = key_pos[None, :kmax] <= qi[:, None]
            if drop:
                mk = mk & key_valid[None, :kmax]
            if gate_beta is not None:
                # confidence-gated transport: additive logit bias beta on transported (halted) keys
                fm = torch.zeros(mk.shape, device=self.device, dtype=self.dtype)
                fm = fm.masked_fill(halt_mask[None, :kmax].expand_as(mk), float(gate_beta))
                mk = fm.masked_fill(~mk, float("-inf"))
            masks.append((s, s + qi.shape[0], kmax, mk[None, None]))

        for L in range(d, g.num_layers):
            x = self.ln_in(L, h_act)
            q, k, v = self.qkv(L, x, cos_a, sin_a)
            Kb, Vb = cache.k[L], cache.v[L]
            Kb[:, :, keep_idx] = k
            Vb[:, :, keep_idx] = v
            if policy == "stale":
                Kb[:, :, halt_idx] = stale_k
                Vb[:, :, halt_idx] = stale_v
            elif drop:
                Kb[:, :, halt_idx] = 0
                Vb[:, :, halt_idx] = 0
            elif policy == "oracle":
                # upper bound / correctness probe: true dense K/V of the halted tokens
                Kb[:, :, halt_idx] = self.oracle_cache.k[L][:, :, halt_idx]
                Vb[:, :, halt_idx] = self.oracle_cache.v[L][:, :, halt_idx]
            else:
                kh, vh = self.transport.halted_kv(policy, L, h_frozen, cos_h, sin_h)
                Kb[:, :, halt_idx] = kh
                Vb[:, :, halt_idx] = vh
            if mode == "compact":
                out = self.attend(q, k, v, is_causal=True)
            elif mode == "flex":
                out = self._flex_attend(q, Kb[:, :, :n], Vb[:, :, :n], block_mask)
            elif len(masks) == 1:
                s, e, kmax, mk = masks[0]
                out = self.attend(q, Kb[:, :, :kmax], Vb[:, :, :kmax], mask=mk)
            else:
                outs = []
                for s, e, kmax, mk in masks:
                    outs.append(self.attend(q[:, :, s:e], Kb[:, :, :kmax], Vb[:, :, :kmax], mask=mk))
                out = torch.cat(outs, dim=2)
            h_act = self.finish_layer(L, h_act, out)
            if drop:
                cache.valid[L] = key_valid

        cache.length = n
        res.last_logits = self.logits_last(h_act[:, -1, :])   # last token is always kept
        res.halt_mask = halt_mask
        res.timings = timings
        return res

    # ------------------------------------------------------------------ extend / decode
    @torch.no_grad()
    def extend(self, cache: KVCache, ids: torch.Tensor) -> torch.Tensor:
        """Append ids (1, m) to cache (teacher forcing / decode). Returns logits (1, m, V) float32."""
        g = self.g
        m = ids.shape[1]
        start = cache.length
        end = start + m
        if end > cache.capacity:
            raise RuntimeError("KV cache capacity exceeded")
        self._check_len(end)
        cos_all, sin_all = self.rope(end)
        cos, sin = cos_all[start:end], sin_all[start:end]
        h = self.lm.embed_tokens(ids)
        key_pos = torch.arange(end, device=self.device)
        q_pos = torch.arange(start, end, device=self.device)
        causal = None
        if m > 1:
            causal = key_pos[None, :] <= q_pos[:, None]
        for L in range(g.num_layers):
            x = self.ln_in(L, h)
            q, k, v = self.qkv(L, x, cos, sin)
            cache.k[L][:, :, start:end] = k
            cache.v[L][:, :, start:end] = v
            mask = causal
            valid = cache.valid[L]
            if valid is not None:
                if valid.shape[0] < end:
                    valid = torch.cat([valid, torch.ones(end - valid.shape[0], dtype=torch.bool,
                                                         device=self.device)])
                    cache.valid[L] = valid
                vm = valid[None, :end]
                mask = vm if mask is None else (mask & vm)
            if mask is not None:
                mask = mask.expand(m, end)[None, None]
            out = self.attend(q, cache.k[L][:, :, :end], cache.v[L][:, :, :end], mask=mask)
            h = self.finish_layer(L, h, out)
        cache.length = end
        return self.lm.lm_head(self.lm.final_norm(h)).float()

    @torch.no_grad()
    def continuation_nll(self, res: PrefillResult, cont_ids: List[int]) -> Optional[float]:
        """Mean teacher-forced NLL of cont_ids given the prefilled cache (cache length restored after)."""
        if not cont_ids:
            return None
        cache = res.cache
        if cache.length + len(cont_ids) > cache.capacity:
            return None
        saved = cache.length
        lp0 = torch.log_softmax(res.last_logits[0], -1)[cont_ids[0]]
        nll = [-float(lp0)]
        if len(cont_ids) > 1:
            ids = torch.tensor([cont_ids[:-1]], device=self.device)
            logits = self.extend(cache, ids)[0]
            lps = torch.log_softmax(logits, -1)
            tgt = torch.tensor(cont_ids[1:], device=self.device)
            nll += (-lps.gather(1, tgt[:, None])[:, 0]).tolist()
        cache.length = saved
        return float(sum(nll) / len(nll))

    @torch.no_grad()
    def greedy_decode(self, res: PrefillResult, max_new: int, eos_ids) -> Tuple[List[int], float]:
        cache = res.cache
        tok = int(torch.argmax(res.last_logits[0]))
        out = [tok]
        sync(self.device)
        t0 = time.perf_counter()
        while len(out) < max_new and tok not in eos_ids and cache.length < cache.capacity:
            logits = self.extend(cache, torch.tensor([[tok]], device=self.device))
            tok = int(torch.argmax(logits[0, -1]))
            out.append(tok)
        sync(self.device)
        return out, time.perf_counter() - t0

    # ------------------------------------------------------------------ HF reference
    @torch.no_grad()
    def hf_last_logits(self, ids: torch.Tensor) -> torch.Tensor:
        try:
            out = self.lm.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.lm.model(input_ids=ids, use_cache=False)
        return out.logits[:, -1, :].float()
