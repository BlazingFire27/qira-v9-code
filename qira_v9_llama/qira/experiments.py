"""
Experiment drivers.  Every measurement row is appended to results/<model>/<exp>.jsonl and
resumed automatically.

  selftest        engine == HF forward; oracle-K/V sparse path == dense; decode / NLL consistency
  calibrate       closed-form transport + thresholds + learned QEI weights per calibration source
  main            matched-budget comparison: Dense, QIRA, QuickSilver, LazyLLM, SlimInfer, KVSimilarity
  transport       QEI selector x {stale, drop, reproject, ridge_plain, ridge_layer, ridge, mlp, oracle}
  selector        {QEI variants, cosine, norm-delta, drift, last-attn, rel-norm, random} x ridge transport
  calib_transfer  QIRA with transport/weights calibrated on idbench | natural | code
  operating       QIRA threshold mode (adaptive per-prompt rate) at Conservative/Balanced/Aggressive
  scaling         needle retrieval at 2K..32K: prefill latency (median of repeats), TTFT, memory
"""

import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from qira.config import MAIN_METHODS, MAX_NEW_TOKENS, METHOD_REGISTRY, PROTOCOL_VERSION
from qira.engine import HaltRequest, sync
from qira.scoring import prefill_flops, rouge_l, score
from qira.store import ResultStore

# ridge_gated is still available (--transport-ablation) but not run by default: on Qwen2.5-7B its
# calibrated logit bias was beta = 0, i.e. identical to ridge.
TRANSPORT_ABLATION = ["stale", "drop", "reproject", "ridge_plain", "ridge_layer", "ridge", "mlp", "oracle"]
SELECTOR_ABLATION = ["qei", "qei_equal", "qei_no_purity", "qei_no_rank", "qei_no_gap", "qei_no_stability",
                     "qei_exact", "qei_fit", "learned", "cosine", "normdelta", "drift", "lastattn", "relnorm",
                     "random"]


class Runner:
    def __init__(self, lm, engine, selectors, formatter, cfg, preset, out_dir: Path, args):
        self.lm, self.e, self.sel, self.fmt = lm, engine, selectors, formatter
        self.cfg, self.preset, self.out = cfg, preset, out_dir
        self.args = args
        self.model = lm.geom.name
        self.eos = lm.eos_ids()
        self.calib: Dict[str, Dict] = {}       # source -> {"bank", "record"}
        self.source: Optional[str] = None
        self._ids_cache: Dict[str, List[int]] = {}
        self.dense_cache: Dict[str, Dict] = {}
        if getattr(self.args, "load_dense", None):
            p = Path(self.args.load_dense)
            if p.exists():
                print(f"[runner] loading cached dense references from {p} ...")
                self.dense_cache = torch.load(p, map_location="cpu")
                print(f"[runner] loaded {len(self.dense_cache)} dense references")
            else:
                print(f"[runner] warning: {p} does not exist!")

    # ------------------------------------------------------------------ calibration state
    def use_source(self, source: str):
        c = self.calib[source]
        self.e.transport = c["bank"]
        self.sel.learned_models = c["record"].get("learned_selectors", {})
        self.source = source

    def threshold(self, selector: str, rate: float):
        tab = self.calib[self.source]["record"]["thresholds"][selector]
        key = min(tab.keys(), key=lambda k: abs(float(k) - rate))
        return tab[key]["tau"], tab[key]["tau_low"]

    # ------------------------------------------------------------------ helpers
    def ids(self, ex, max_context=None) -> List[int]:
        k = f"{ex['id']}|{max_context}"
        if k not in self._ids_cache:
            self._ids_cache[k] = self.fmt.encode(ex, max_context=max_context)
        return self._ids_cache[k]

    def _mem_reset(self):
        if self.e.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.e.device)

    def _mem_peak(self):
        if self.e.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.e.device) / 2 ** 30
        return None

    def _timed_prefill(self, ids_t, req, extra, keep_decision=False):
        sync(self.e.device)
        self._mem_reset()
        t0 = time.perf_counter()
        res = self.e.prefill(ids_t, req, extra_capacity=extra, keep_decision=keep_decision)
        _ = int(torch.argmax(res.last_logits[0]))          # first token selected => TTFT
        sync(self.e.device)
        return res, time.perf_counter() - t0, self._mem_peak()

    def max_new(self, task):
        if self.args.max_new_tokens:
            return min(self.args.max_new_tokens, MAX_NEW_TOKENS[task])
        return MAX_NEW_TOKENS[task]

    # ------------------------------------------------------------------ dense reference
    def dense_reference(self, ex, store: ResultStore, exp: str, gen: bool = True, max_context=None) -> Dict:
        ex_id = ex["id"]
        key = ResultStore.make_key(exp, "Dense", "dense", 0.0, ex_id)
        if ex_id in self.dense_cache:
            cached = self.dense_cache[ex_id]
            logp = cached["logp"].to(device=self.e.device, dtype=self.e.dtype)
            ref = dict(logp=logp, top1=cached["top1"], cache=None, n=cached["n"],
                       ans_nll=cached.get("ans_nll"), text=cached.get("text"),
                       prefill_s=cached.get("prefill_s", 0.2))
            if not store.done(key):
                sc = {"quality": cached["score"]} if isinstance(cached.get("score"), (int, float)) else cached.get("score")
                row = self._row(exp, ex, "Dense", None, None, "dense", 0.0, cached["n"], 0,
                                ref["prefill_s"], None, None, None, ref["text"], sc,
                                kl=0.0, top1=1.0, ans_nll=ref["ans_nll"], ans_ref=ref["ans_nll"], fid=1.0,
                                flops=prefill_flops(self.lm.geom, cached["n"]), pairs=None)
                store.add(key, row)
            return ref

        ids_l = self.ids(ex, max_context)
        ids_t = torch.tensor([ids_l], device=self.e.device)
        task = ex["task"]
        mx = self.max_new(task)
        ans = self.fmt.answer_ids(ex)
        extra = mx + len(ans) + 8
        self.e.oracle_cache = None
        res, t_pre, mem = self._timed_prefill(ids_t, None, extra)
        logp = torch.log_softmax(res.last_logits[0], -1)
        ref = dict(logp=logp, top1=int(torch.argmax(logp)), cache=res.cache, n=len(ids_l))
        ref["ans_nll"] = self.e.continuation_nll(res, ans) if ans else None
        row = store.get(key)
        if row is None and gen:
            hf_t = None
            if self.args.hf_timing:
                sync(self.e.device)
                t0 = time.perf_counter()
                hf_logits = self.e.hf_last_logits(ids_t)
                sync(self.e.device)
                hf_t = time.perf_counter() - t0
                hf_kl = float((logp.exp() * (logp - torch.log_softmax(hf_logits[0], -1))).sum())
            else:
                hf_kl = None
            toks, dec_t = self.e.greedy_decode(res, mx, self.eos)
            text = self.lm.tokenizer.decode(toks, skip_special_tokens=True)
            sc = score(task, text, ex["reference"])
            n = len(ids_l)
            row = self._row(exp, ex, "Dense", None, None, "dense", 0.0, n, 0, t_pre, mem, dec_t, toks, text, sc,
                            kl=0.0, top1=1.0, ans_nll=ref["ans_nll"], ans_ref=ref["ans_nll"], fid=1.0,
                            flops=prefill_flops(self.lm.geom, n), pairs=None)
            row.update(hf_prefill_s=hf_t, engine_vs_hf_kl=hf_kl)
            store.add(key, row)
        ref["text"] = row["prediction"] if row else None
        ref["prefill_s"] = row["prefill_s"] if row else t_pre
        # the dense cache doubles as the oracle K/V source
        self.e.oracle_cache = res.cache
        return ref

    # ------------------------------------------------------------------ one measurement
    def _row(self, exp, ex, method, selector, transport, mode, value, n, n_halt, t_pre, mem, dec_t, toks,
             text, sc, kl, top1, ans_nll, ans_ref, fid, flops, pairs, eligible=None, extra=None):
        dense_fl = prefill_flops(self.lm.geom, n)
        g = self.lm.geom
        row = dict(
            protocol=PROTOCOL_VERSION, exp=exp, model=self.model, task=ex["task"], sample_id=ex["id"],
            subset=ex.get("meta", {}).get("subset"), redundancy_pct=ex.get("meta", {}).get("redundancy_pct"),
            target_len=ex.get("meta", {}).get("target_len"),
            method=method, selector=selector, transport=transport, mode=mode, budget=float(value),
            calib_source=self.source if selector else None,
            prompt_tokens=n, eligible_tokens=eligible, halted_tokens=int(n_halt),
            halt_frac=n_halt / max(1, n),
            token_layer_reduction=n_halt * (g.num_layers - g.decision_layer) / max(1, n * g.num_layers),
            flop_reduction=1.0 - flops / dense_fl if dense_fl > 0 else 0.0,
            prefill_s=t_pre, ttft_s=t_pre, decode_s=dec_t, gen_tokens=len(toks) if toks else 0,
            decode_tps=(len(toks) - 1) / dec_t if toks and dec_t and len(toks) > 1 else None,
            prefill_tok_per_s=n / t_pre if t_pre else None, peak_mem_gb=mem,
            quality=(sc.get("quality") if isinstance(sc, dict) else sc) if sc is not None else None,
            em=(sc.get("em") if isinstance(sc, dict) else None) if sc is not None else None,
            kl_last=kl, top1_agree=top1, ans_nll=ans_nll,
            ans_nll_delta=(ans_nll - ans_ref) if (ans_nll is not None and ans_ref is not None) else None,
            fidelity_rougeL_vs_dense=fid, prediction=text,
        )
        if extra:
            row.update(extra)
        return row

    def run_one(self, exp: str, store: ResultStore, ex: Dict, ref: Dict, method: str, selector: str,
                transport: str, mode: str, value: float, gen: bool = True, max_context=None,
                jaccard_with: Optional[List[str]] = None, timing_repeats: int = 1, tag: str = "") -> Optional[Dict]:
        key = ResultStore.make_key(exp, method, mode, round(float(value), 4), self.source if selector else "-",
                                   ex["id"], tag)
        if store.done(key):
            return None
        ids_l = self.ids(ex, max_context)
        ids_t = torch.tensor([ids_l], device=self.e.device)
        n = len(ids_l)
        task = ex["task"]
        mx = self.max_new(task) if gen else 1
        ans = self.fmt.answer_ids(ex)
        extra_cap = mx + len(ans) + 8
        tau = tau_low = None
        if mode == "threshold":
            tau, tau_low = self.threshold(selector, value)
        req = HaltRequest(selector=selector, transport=transport, mode=mode, value=value, tau=tau, tau_low=tau_low)
        times = []
        res = mem = None
        for r in range(max(1, timing_repeats)):
            if res is not None:
                del res
            res, t_pre, mem = self._timed_prefill(ids_t, req, extra_cap, keep_decision=bool(jaccard_with))
            times.append(t_pre)
        t_pre = float(np.median(times))
        n_halt = int(res.halt_mask.sum()) if res.halt_mask is not None else 0
        logp = torch.log_softmax(res.last_logits[0], -1)
        kl = float((ref["logp"].float().exp() * (ref["logp"].float() - logp.float())).sum())
        top1 = float(int(torch.argmax(logp)) == ref["top1"])
        ans_nll = self.e.continuation_nll(res, ans) if ans else None
        extra = {}
        if jaccard_with and res.decision_state is not None and res.halt_mask is not None:
            h, hp, cos, sin = res.decision_state
            lo, hi = self.e.eligible_range(n)
            for other in jaccard_with:
                s2 = self.sel.score(other, h, hp, cos, sin)
                m2 = self.sel.select(s2, n, lo, hi, HaltRequest(other, transport, "budget", value))
                inter = int((m2 & res.halt_mask).sum())
                union = int((m2 | res.halt_mask).sum())
                extra[f"jaccard_{other}"] = inter / union if union else 1.0
            res.decision_state = None
        toks, dec_t, text, sc, fid = None, None, None, None, None
        if gen:
            toks, dec_t = self.e.greedy_decode(res, mx, self.eos)
            text = self.lm.tokenizer.decode(toks, skip_special_tokens=True)
            sc = score(task, text, ex["reference"])
            fid = rouge_l(text, ref["text"]) if ref.get("text") is not None else None
        fl = prefill_flops(self.lm.geom, n, n_halt, res.attn_pairs_after, transport, self.cfg.mlp_hidden)
        if timing_repeats > 1:
            extra["prefill_s_all"] = [round(t, 6) for t in times]
        if res.timings.get("select_s") is not None:
            extra["select_s"] = res.timings["select_s"]
        if res.timings.get("attn_mode") is not None:
            extra["attn_mode"] = res.timings["attn_mode"]
        lo, hi = self.e.eligible_range(n)
        row = self._row(exp, ex, method, selector, transport, mode, value, n, n_halt, t_pre, mem, dec_t, toks,
                        text, sc, kl, top1, ans_nll, ref.get("ans_nll"), fid, fl, res.attn_pairs_after,
                        eligible=hi - lo, extra=extra)
        store.add(key, row)
        del res
        return row

    # ------------------------------------------------------------------ experiments
    def _loop(self, exp: str, data: Dict[str, List[Dict]], plan, gen=True, max_context=None, desc=""):
        store = ResultStore(self.out / f"{exp}.jsonl")
        total = sum(len(v) for v in data.values())
        done = 0
        t_start = time.time()
        for task, exs in data.items():
            for ex in exs:
                todo = [p for p in plan(ex) if not store.done(ResultStore.make_key(
                    exp, p["method"], p["mode"], round(float(p["value"]), 4),
                    (p.get("source") or self.source) if p.get("selector") else "-", ex["id"], p.get("tag", "")))]
                need_dense = not store.done(ResultStore.make_key(exp, "Dense", "dense", 0.0, ex["id"]))
                if todo or need_dense:
                    ref = self.dense_reference(ex, store, exp, gen=gen, max_context=max_context)
                    for p in todo:
                        if p.get("source") and p["source"] != self.source:
                            self.use_source(p["source"])
                        try:
                            self.run_one(exp, store, ex, ref, p["method"], p["selector"], p["transport"],
                                         p["mode"], p["value"], gen=gen, max_context=max_context,
                                         jaccard_with=p.get("jaccard"), tag=p.get("tag", ""))
                        except torch.cuda.OutOfMemoryError:
                            print(f"[{exp}] OOM on {ex['id']} {p['method']}@{p['value']}; skipped")
                            torch.cuda.empty_cache()
                    self.e.oracle_cache = None
                    del ref
                done += 1
                if done % 5 == 0 or done == total:
                    el = time.time() - t_start
                    print(f"[{exp}] {done}/{total} samples  ({el / 60:.1f} min elapsed)", flush=True)
        if self.e.device.type == "cuda":
            torch.cuda.empty_cache()
        return store

    def exp_dense_only(self, data: Dict[str, List[Dict]], out_file: Path):
        """Precompute and cache Dense reference for all examples with resumable checkpointing."""
        store = ResultStore(self.out / "Dense.jsonl")
        dense_dict = {}
        if out_file.exists():
            try:
                dense_dict = torch.load(out_file, map_location="cpu")
                print(f"[dense-only] Resuming from existing checkpoint: {len(dense_dict)} references loaded from {out_file}", flush=True)
            except Exception as e:
                print(f"[dense-only] Warning: could not load existing {out_file}: {e}")
                dense_dict = {}

        def _save_checkpoint():
            if dense_dict:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_out = out_file.with_suffix(".pt.tmp")
                torch.save(dense_dict, tmp_out)
                tmp_out.replace(out_file)

        all_exs = [ex for exs in data.values() for ex in exs]
        total = len(all_exs)
        print(f"[dense-only] Pre-computing Dense reference for {total} examples ...", flush=True)
        t0 = time.time()
        new_since_save = 0
        try:
            for i, ex in enumerate(all_exs):
                ex_id = ex["id"]
                key = ResultStore.make_key("main", "Dense", "dense", 0.0, ex_id)
                if ex_id in dense_dict and store.done(key):
                    continue

                ref = self.dense_reference(ex, store, "main", gen=True)
                row = store.get(key)
                dense_dict[ex_id] = dict(
                    logp=ref["logp"].half().cpu(),
                    top1=ref["top1"],
                    n=ref["n"],
                    ans_nll=ref["ans_nll"],
                    text=ref["text"],
                    prefill_s=ref["prefill_s"],
                    score=row.get("quality") if row else None,
                )
                new_since_save += 1

                # Periodic checkpoint save every 100 newly processed samples
                if new_since_save >= 100:
                    _save_checkpoint()
                    new_since_save = 0

                if (i + 1) % 25 == 0 or (i + 1) == total:
                    el = time.time() - t0
                    pct = ((i + 1) / total) * 100.0
                    print(f"[dense-only] {i + 1}/{total} ({pct:.1f}%) in {el/60:.1f} min", flush=True)
        finally:
            _save_checkpoint()
            print(f"[dense-only] Saved {len(dense_dict)} dense references to {out_file}", flush=True)
        return store

    def exp_main(self, data, methods: List[str], budgets: List[float]):
        def plan(ex):
            out = []
            for b in budgets:
                for m in methods:
                    if m == "Dense":
                        continue
                    sel, tr = METHOD_REGISTRY[m]
                    out.append(dict(method=m, selector=sel, transport=tr, mode="budget", value=b))
            return out
        return self._loop("main", data, plan)

    def exp_transport(self, data, budgets):
        def plan(ex):
            return [dict(method=f"qei+{t}", selector="qei", transport=t, mode="budget", value=b)
                    for b in budgets for t in TRANSPORT_ABLATION]
        return self._loop("transport", data, plan)

    def exp_selector(self, data, budgets):
        def plan(ex):
            return [dict(method=f"{s}+ridge", selector=s, transport="ridge", mode="budget", value=b,
                         jaccard=["cosine", "normdelta", "drift"] if s.startswith("qei") else None)
                    for b in budgets for s in SELECTOR_ABLATION]
        return self._loop("selector", data, plan)

    def exp_transfer(self, data, budgets, sources):
        default = self.source

        def plan(ex):
            return [dict(method=f"QIRA[{src}]", selector="qei", transport="ridge", mode="budget", value=b,
                         source=src, tag=src)
                    for src in sources for b in budgets]
        try:
            return self._loop("calib_transfer", data, plan)
        finally:
            self.use_source(default)

    def exp_operating(self, data, points: Dict[str, float]):
        def plan(ex):
            return [dict(method=f"QIRA-{name}", selector="qei", transport="ridge", mode="threshold", value=r)
                    for name, r in points.items()]
        return self._loop("operating", data, plan)

    def exp_latency(self, examples: List[Dict], methods: List[str], budgets: List[float], repeats: int):
        """Clean wall-clock prefill timing on long prompts: warm-up + median of `repeats`, no generation."""
        exp = "latency"
        store = ResultStore(self.out / f"{exp}.jsonl")
        for i, ex in enumerate(examples):
            ids_t = torch.tensor([self.ids(ex)], device=self.e.device)
            dkey = ResultStore.make_key(exp, "Dense", "dense", 0.0, ex["id"])
            plans = [(m, b) for b in budgets for m in methods]
            todo = [(m, b) for (m, b) in plans if not store.done(ResultStore.make_key(
                exp, m, "budget", round(b, 4), self.source, ex["id"], ""))]
            if store.done(dkey) and not todo:
                continue
            try:
                ref = self.dense_reference(ex, store, exp, gen=False)
                if not store.done(dkey):
                    self.e.prefill(ids_t, None)
                    times, mem = [], None
                    for _ in range(repeats):
                        res, t, mem = self._timed_prefill(ids_t, None, 8)
                        times.append(t)
                        del res
                    n = ids_t.shape[1]
                    row = self._row(exp, ex, "Dense", None, None, "dense", 0.0, n, 0, float(np.median(times)), mem,
                                    None, None, None, None, 0.0, 1.0, None, None, None,
                                    prefill_flops(self.lm.geom, n), None)
                    row["prefill_s_all"] = [round(t, 6) for t in times]
                    store.add(dkey, row)
                for (m, b) in todo:
                    sel, tr = METHOD_REGISTRY[m]
                    self.e.prefill(ids_t, HaltRequest(sel, tr, "budget", b))          # warm-up
                    self.run_one(exp, store, ex, ref, m, sel, tr, "budget", b, gen=False, timing_repeats=repeats)
                self.e.oracle_cache = None
                del ref
            except torch.cuda.OutOfMemoryError:
                print(f"[latency] OOM at {ex['id']}; skipping")
                self.e.oracle_cache = None
                torch.cuda.empty_cache()
            if (i + 1) % 5 == 0:
                print(f"[latency] {i + 1}/{len(examples)} samples", flush=True)
        return store

    def exp_scaling(self, examples: List[Dict], budgets: List[float], repeats: int, baselines_at: float):
        exp = "scaling"
        store = ResultStore(self.out / f"{exp}.jsonl")
        for ex in examples:
            T = ex["meta"]["target_len"]
            mc = min(T + 256, self.lm.geom.max_positions - 32)
            plans = [("QIRA", "qei", "ridge", b) for b in budgets]
            plans += [("QuickSilver", "drift", "stale", baselines_at), ("LazyLLM", "lastattn", "drop", baselines_at)]
            try:
                ref = None
                dkey = ResultStore.make_key(exp, "Dense", "dense", 0.0, ex["id"])
                if not store.done(dkey):
                    # dense: engine + HF reference, timed with repeats after one warm-up
                    ids_t = torch.tensor([self.ids(ex, mc)], device=self.e.device)
                    self.e.prefill(ids_t, None)
                    times, hf_times = [], []
                    for _ in range(repeats):
                        res, t, mem = self._timed_prefill(ids_t, None, 32)
                        times.append(t)
                        del res
                    for _ in range(repeats):
                        sync(self.e.device)
                        self._mem_reset()
                        t0 = time.perf_counter()
                        self.e.hf_last_logits(ids_t)
                        sync(self.e.device)
                        hf_times.append(time.perf_counter() - t0)
                    hf_mem = self._mem_peak()
                    ref = self.dense_reference(ex, store, exp, gen=False, max_context=mc)
                    res, _, mem = self._timed_prefill(ids_t, None, 32)
                    toks, dec_t = self.e.greedy_decode(res, self.max_new("needle"), self.eos)
                    text = self.lm.tokenizer.decode(toks, skip_special_tokens=True)
                    n = ids_t.shape[1]
                    row = self._row(exp, ex, "Dense", None, None, "dense", 0.0, n, 0, float(np.median(times)), mem,
                                    dec_t, toks, text, score("needle", text, ex["reference"]), 0.0, 1.0,
                                    ref.get("ans_nll"), ref.get("ans_nll"), 1.0, prefill_flops(self.lm.geom, n), None)
                    row.update(hf_prefill_s=float(np.median(hf_times)), hf_peak_mem_gb=hf_mem,
                               prefill_s_all=[round(t, 6) for t in times])
                    store.add(dkey, row)
                    del res
                for (m, s, t, b) in plans:
                    key = ResultStore.make_key(exp, m, "budget", round(b, 4), self.source, ex["id"], "")
                    if store.done(key):
                        continue
                    if ref is None:
                        ref = self.dense_reference(ex, store, exp, gen=False, max_context=mc)
                        ref["text"] = store.get(dkey)["prediction"]
                    ids_t = torch.tensor([self.ids(ex, mc)], device=self.e.device)
                    self.e.prefill(ids_t, HaltRequest(s, t, "budget", b))      # warm-up
                    self.run_one(exp, store, ex, ref, m, s, t, "budget", b, gen=True, max_context=mc,
                                 timing_repeats=repeats)
                self.e.oracle_cache = None
                del ref
            except torch.cuda.OutOfMemoryError:
                print(f"[scaling] OOM at {ex['id']}; skipping")
                self.e.oracle_cache = None
            gc.collect()
            if self.e.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[scaling] done {ex['id']} (n_target={T})", flush=True)
        return store
