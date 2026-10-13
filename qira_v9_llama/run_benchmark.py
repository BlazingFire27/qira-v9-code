#!/usr/bin/env python3
"""
QIRA v8 benchmark — single entry point.

    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset quick            # ~1-2 h smoke run
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --preset paper            # full study
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --experiments report      # rebuild report only

Experiments (default "all"): selftest calibrate main transport selector calib_transfer operating scaling report
Every measurement is appended to results/<model>/<experiment>.jsonl; re-running resumes.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from qira.config import (CALIB_SOURCES, CODE_VERSION, LATENCY_METHODS, LONGBENCH_ALL_SUBSETS, LONGBENCH_EVAL_SUBSETS, MAIN_METHODS,
                         METHOD_REGISTRY, PRESETS, PROTOCOL_VERSION, QIRAConfig, TASKS)
from qira.models import LoadedModel
from qira.engine import Engine, HaltRequest, sync
from qira.selectors import SelectorBank
from qira.tasks import (LongBench, PromptFormatter, build_calibration_pool, build_idbench, build_needle_examples,
                        load_gsm8k, load_humaneval, load_longbench_eval, load_mmlu, load_squad, load_sst2)
from qira.calibration import load_calibration, run_calibration
from qira.experiments import Runner
from qira.report import build_report

ALL_EXPERIMENTS = ["selftest", "calibrate", "main", "transport", "selector", "calib_transfer", "operating",
                   "latency", "scaling", "report"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_hash(cfg: QIRAConfig, extra: dict) -> str:
    s = json.dumps(dict(cfg=asdict(cfg), **extra), sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- selftest

@torch.no_grad()
def selftest(runner: Runner, examples, out_dir: Path, post_calibration: bool = False) -> bool:
    e, lm = runner.e, runner.lm
    report = []
    ok = True
    policies = ["stale", "drop", "reproject"] + (["ridge_plain", "ridge_layer", "ridge", "ridge_gated", "mlp"] if post_calibration else [])
    for ex in examples:
        ids_l = runner.fmt.encode(ex, max_context=1024)
        ids = torch.tensor([ids_l], device=e.device)
        n = len(ids_l)
        rec = dict(sample=ex["id"], n=n)
        dense = e.prefill(ids, None, extra_capacity=64)
        lp_e = torch.log_softmax(dense.last_logits[0], -1)
        if not post_calibration:
            lp_h = torch.log_softmax(e.hf_last_logits(ids)[0], -1)
            rec["engine_vs_hf_kl"] = float((lp_h.exp() * (lp_h - lp_e)).sum())
            rec["engine_vs_hf_top1"] = bool(int(lp_h.argmax()) == int(lp_e.argmax()))
            # decode consistency
            toks, _ = e.greedy_decode(dense, 16, set())
            try:
                hf = lm.model.generate(ids, max_new_tokens=16, do_sample=False, temperature=None, top_p=None,
                                       top_k=None, pad_token_id=lm.tokenizer.pad_token_id, eos_token_id=None)
            except TypeError:
                hf = lm.model.generate(ids, max_new_tokens=16, do_sample=False)
            hf_toks = hf[0, n:].tolist()
            match = 0
            for a, b in zip(toks, hf_toks):
                if a != b:
                    break
                match += 1
            rec["decode_prefix_match_of_16"] = match
            # teacher-forced NLL consistency
            ans = runner.fmt.answer_ids(ex) or lm.tokenizer(" the answer", add_special_tokens=False)["input_ids"]
            dense2 = e.prefill(ids, None, extra_capacity=len(ans) + 4)
            nll_e = e.continuation_nll(dense2, ans)
            full = torch.tensor([ids_l + ans], device=e.device)
            logits = lm.model(input_ids=full, use_cache=False).logits[0, n - 1:-1].float()
            nll_h = float(-torch.log_softmax(logits, -1).gather(1, torch.tensor(ans, device=e.device)[:, None]).mean())
            rec["nll_engine"], rec["nll_hf"] = nll_e, nll_h
            if rec["engine_vs_hf_kl"] > 5e-2 or abs(nll_e - nll_h) > 0.1 * max(1.0, abs(nll_h)):
                ok = False
        # oracle K/V => active tokens must reproduce dense exactly (sparse-path correctness)
        dense3 = e.prefill(ids, None)
        e.oracle_cache = dense3.cache
        orc = e.prefill(ids, HaltRequest("qei", "oracle", "budget", 0.5))
        lp_o = torch.log_softmax(orc.last_logits[0], -1)
        rec["oracle_kl"] = float((lp_e.exp() * (lp_e - lp_o)).sum())
        rec["oracle_halted"] = int(orc.halt_mask.sum()) if orc.halt_mask is not None else 0
        if rec["oracle_kl"] > 2e-2:
            ok = False
        e.oracle_cache = None
        for pol in policies:
            r = e.prefill(ids, HaltRequest("qei", pol, "budget", 0.5), extra_capacity=8)
            lp = torch.log_softmax(r.last_logits[0], -1)
            rec[f"kl_{pol}@0.5"] = float((lp_e.exp() * (lp_e - lp)).sum())
            if not torch.isfinite(r.last_logits).all():
                ok = False
                rec[f"nan_{pol}"] = True
            # decode a few steps through the sparse cache (exercises drop masks etc.)
            e.greedy_decode(r, 4, set())
        report.append(rec)
        print("[selftest]", json.dumps(rec))
    name = "selftest_post_calibration.json" if post_calibration else "selftest.json"
    (out_dir / name).write_text(json.dumps(dict(passed=ok, checks=report), indent=1))
    print(f"[selftest] {'PASSED' if ok else 'FAILED'} -> {out_dir / name}")
    return ok


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="QIRA v8: QEI halting + cross-layer K/V transport benchmark")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Hugging Face model ID (default: meta-llama/Llama-3.1-8B-Instruct; auto-falls back to NousResearch if gated)")
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--experiments", nargs="+", default=["all"], help=" ".join(ALL_EXPERIMENTS))
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    ap.add_argument("--budgets", nargs="+", type=float, default=None, help="override main-experiment budgets")
    ap.add_argument("--calib-source", default="natural", choices=CALIB_SOURCES)
    ap.add_argument("--transfer-sources", nargs="+", default=CALIB_SOURCES)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--no-mlp", action="store_true", help="skip the residual-MLP transport variant")
    ap.add_argument("--calib-prompts", type=int, default=None, help="override calibration prompts per source")
    ap.add_argument("--calib-max-tokens", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="cap samples per task")
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="cap generation length (debug)")
    ap.add_argument("--scaling-lengths", nargs="+", type=int, default=None)
    ap.add_argument("--decision-layer-frac", type=float, default=0.5)
    ap.add_argument("--protect-suffix", type=int, default=16)
    ap.add_argument("--attn-chunk", type=int, default=4096)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-hf-timing", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force-restart", action="store_true", help="delete this model's result files first")
    ap.add_argument("--ignore-selftest", action="store_true")
    ap.add_argument("--attn-backend", default="auto", choices=["auto", "flex", "mask"],
                    help="sparse-attention kernel for halted prefill (auto = FlexAttention on CUDA if it validates)")
    ap.add_argument("--reset-experiments", nargs="+", default=[],
                    help="delete these experiments' result files before running (e.g. scaling latency)")
    ap.add_argument("--dense-only", action="store_true",
                    help="run Dense pre-computation for all requested samples, save dense cache, and exit")
    ap.add_argument("--load-dense", type=str, default=None,
                    help="path to pre-computed dense_references.pt")
    ap.add_argument("--longbench-all", action="store_true",
                    help="evaluate all 16 LongBench tasks (all 3,750 samples)")
    ap.add_argument("--shard-idx", type=int, default=0, help="shard index for multi-GPU data splitting (0-indexed)")
    ap.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    args = ap.parse_args()
    args.hf_timing = not args.no_hf_timing

    exps = ALL_EXPERIMENTS if "all" in args.experiments else args.experiments
    bad = [x for x in exps if x not in ALL_EXPERIMENTS]
    if bad:
        ap.error(f"unknown experiments {bad}")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if exps == ["report"]:
        from qira.models import model_short_name
        out = Path(args.output_dir) / model_short_name(args.model)
        meta = json.loads((out / "run_config.json").read_text()) if (out / "run_config.json").exists() else {}
        build_report(out, out.name, out / "calibration" / meta.get("calib_source", args.calib_source), meta)
        return
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]
    preset = PRESETS[args.preset]
    if args.calib_prompts:
        preset = replace(preset, calib_prompts={k: args.calib_prompts for k in preset.calib_prompts})
    if args.calib_max_tokens:
        preset = replace(preset, calib_max_tokens=args.calib_max_tokens)

    cfg = QIRAConfig(decision_layer_fraction=args.decision_layer_frac, protect_suffix=args.protect_suffix,
                     attn_query_chunk=args.attn_chunk)
    lm = LoadedModel(args.model, args.device, dtype, token, cfg.decision_layer_fraction)
    g = lm.geom
    out = Path(args.output_dir) / g.name
    out.mkdir(parents=True, exist_ok=True)
    if args.force_restart:
        for p in out.glob("*.jsonl"):
            p.unlink()
    for x in args.reset_experiments:
        p = out / f"{x}.jsonl"
        if p.exists():
            print(f"[reset] removing {p}")
            p.unlink()
    unknown = [m for m in args.methods if m not in METHOD_REGISTRY]
    if unknown:
        ap.error(f"unknown methods {unknown}; choose from {list(METHOD_REGISTRY)}")
    max_context = min(args.max_context or preset.max_context, g.max_positions)
    engine = Engine(lm, cfg)
    if args.attn_backend == "flex" or (args.attn_backend == "auto" and engine.device.type == "cuda"):
        ok_flex = engine.enable_flex()
        if args.attn_backend == "flex" and not ok_flex:
            sys.exit("--attn-backend flex requested but FlexAttention failed validation")
    sel = SelectorBank(engine, cfg)
    engine.selectors = sel
    from qira.transport import TransportBank
    engine.transport = TransportBank(engine, cfg)       # uncalibrated: supports "reproject" only
    fmt = PromptFormatter(lm.tokenizer, use_chat=not args.no_chat_template, max_context=max_context)
    runner = Runner(lm, engine, sel, fmt, cfg, preset, out, args)
    safe_argv = []
    skip_next = False
    for a in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if a in ("--hf-token", "--token"):
            skip_next = True
            continue
        if a.startswith("--hf-token=") or a.startswith("--token=") or a.startswith("hf_"):
            continue
        safe_argv.append(a)

    meta = dict(protocol=PROTOCOL_VERSION, timestamp=datetime.now().isoformat(), model=args.model,
                preset=preset.name, calib_source=args.calib_source, decision_layer=g.decision_layer,
                num_layers=g.num_layers, chat_template=fmt.use_chat, max_context=max_context,
                dtype=args.dtype, device=args.device, torch=torch.__version__, code_version=CODE_VERSION,
                attn_backend=engine.attn_backend,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                qira_config=asdict(cfg), preset_config=asdict(preset), experiments=exps, argv=safe_argv)
    (out / "run_config.json").write_text(json.dumps(meta, indent=1, default=str))

    lb = LongBench(Path(args.output_dir) / "longbench_cache")
    idb_cal, idb_eval = build_idbench(eval_per_density=preset.idbench_eval_per_density)

    def cap(n):
        return n if args.max_samples is None else min(n, args.max_samples)

    # ---------------- warm-up (kernels, allocator) ----------------
    warm = engine.prefill(torch.tensor([fmt.encode(idb_cal[0])], device=engine.device), None, extra_capacity=4)
    engine.greedy_decode(warm, 3, set())
    del warm

    # ---------------- selftest ----------------
    st_examples = [idb_cal[0]]
    try:
        st_examples.append(build_calibration_pool("natural", lb, 4, idb_cal)[0])
    except Exception as ex_:
        print(f"[WARN] LongBench unavailable for selftest: {ex_}")
    if "selftest" in exps:
        passed = selftest(runner, st_examples, out)
        if not passed and not args.ignore_selftest:
            sys.exit("selftest failed: the manual engine does not reproduce this model. "
                     "Inspect results/<model>/selftest.json (or pass --ignore-selftest).")

    # ---------------- calibration ----------------
    need_sources = {args.calib_source}
    if "calib_transfer" in exps:
        need_sources |= set(args.transfer_sources)
    needs_calib = (not args.dense_only) and any(x in exps for x in ["calibrate", "main", "transport", "selector", "calib_transfer",
                                                                      "operating", "latency", "scaling"])
    h = cfg_hash(cfg, dict(model=args.model, chat=fmt.use_chat, calib_max=preset.calib_max_tokens,
                           calib_n=preset.calib_prompts, mlp=not args.no_mlp, version=PROTOCOL_VERSION + "-c3"))
    if needs_calib:
        for src in sorted(need_sources, key=lambda s: s != args.calib_source):
            cdir = out / "calibration" / src
            stamp = cdir / "stamp.txt"
            if (not args.recalibrate and stamp.exists() and stamp.read_text().strip() == h
                    and (cdir / "transport.pt").exists()):
                print(f"[calibration] reusing {cdir}")
                runner.calib[src] = load_calibration(engine, sel, cfg, cdir)
            else:
                pool = build_calibration_pool(src, lb, preset.calib_prompts[src], idb_cal)
                print(f"[calibration] source={src}: {len(pool)} prompts (max {preset.calib_max_tokens} tokens each)")
                t0 = time.time()
                runner.calib[src] = run_calibration(engine, sel, fmt, pool, cfg, cdir, preset.calib_max_tokens,
                                                    fit_mlp=not args.no_mlp)
                stamp.write_text(h)
                print(f"[calibration] {src} done in {time.time() - t0:.1f}s")
            if engine.device.type == "cuda":
                torch.cuda.empty_cache()
        runner.use_source(args.calib_source)
        # warm-up of the sparse kernels (FlexAttention compiles on first use) before any timing
        wids = torch.tensor([fmt.encode(idb_cal[0])], device=engine.device)
        for pol in ("ridge", "drop", "stale"):
            engine.prefill(wids, HaltRequest("qei", pol, "budget", 0.5))
        if "selftest" in exps:
            ok2 = selftest(runner, st_examples[:1], out, post_calibration=True)
            if not ok2 and not args.ignore_selftest:
                sys.exit("post-calibration selftest failed (non-finite transport output)")

    # ---------------- data ----------------
    def load_tasks(tasks):
        data = {}
        for t in tasks:
            if t not in args.tasks:
                continue
            try:
                if t == "idbench":
                    data[t] = idb_eval[: cap(len(idb_eval))] if args.max_samples else idb_eval
                elif t == "longbench":
                    subsets = LONGBENCH_ALL_SUBSETS if args.longbench_all else LONGBENCH_EVAL_SUBSETS
                    n_lb = 3750 if args.longbench_all else preset.samples["longbench"]
                    data[t] = load_longbench_eval(lb, cap(n_lb), subsets)
                    if args.max_samples:
                        data[t] = data[t][: args.max_samples]
                elif t == "squad":
                    data[t] = load_squad(cap(preset.samples["squad"]))
                elif t == "mmlu":
                    data[t] = load_mmlu(cap(preset.samples["mmlu"]))
                elif t == "gsm8k":
                    data[t] = load_gsm8k(cap(preset.samples["gsm8k"]))
                elif t == "humaneval":
                    data[t] = load_humaneval(cap(preset.samples["humaneval"]), fmt.use_chat)
                elif t in ("sst2", "sst"):
                    data[t] = load_sst2(cap(preset.samples.get("sst2", 872)))
            except Exception as ex_:
                print(f"[WARN] task {t} unavailable: {ex_}")
        if args.num_shards > 1:
            for t in list(data.keys()):
                data[t] = data[t][args.shard_idx :: args.num_shards]
        for t, v in data.items():
            print(f"  task {t}: {len(v)} samples")
        return data

    if args.dense_only:
        dense_out = out / "dense_references.pt"
        print(f"[run_benchmark] Running in --dense-only mode. Saving references to {dense_out} ...")
        runner.exp_dense_only(load_tasks(args.tasks), dense_out)
        print(f"[run_benchmark] Dense-only evaluation complete! Exiting successfully.")
        return

    try:
        if "main" in exps:
            budgets = args.budgets or preset.main_budgets
            runner.exp_main(load_tasks(args.tasks), args.methods, budgets)
        if "transport" in exps:
            runner.exp_transport(load_tasks(preset.transport_tasks), preset.transport_budgets)
        if "selector" in exps:
            runner.exp_selector(load_tasks(preset.selector_tasks), preset.selector_budgets)
        if "calib_transfer" in exps:
            runner.exp_transfer(load_tasks(preset.transfer_tasks), preset.transfer_budgets,
                                [s for s in args.transfer_sources if s in runner.calib])
        if "operating" in exps:
            runner.exp_operating(load_tasks(TASKS), preset.operating_points)
        if "latency" in exps:
            pool = load_longbench_eval(lb, max(preset.latency_samples, 8), LONGBENCH_EVAL_SUBSETS)
            longs = [ex for ex in pool if len(fmt.encode(ex)) >= 2048][: preset.latency_samples]
            print(f"  latency: {len(longs)} LongBench prompts >= 2048 tokens")
            lat_methods = [m for m in LATENCY_METHODS if m in args.methods]
            runner.exp_latency(longs, lat_methods, args.budgets or preset.main_budgets, preset.timing_repeats)
        if "scaling" in exps:
            lengths = [L for L in (args.scaling_lengths or preset.scaling_lengths) if L <= g.max_positions]
            skipped = [L for L in (args.scaling_lengths or preset.scaling_lengths) if L > g.max_positions]
            if skipped:
                print(f"[scaling] skipping lengths {skipped} (> model max_position_embeddings={g.max_positions})")
            needles = build_needle_examples(lb, lm.tokenizer, lengths, preset.scaling_samples, fmt)
            mid = preset.scaling_budgets[len(preset.scaling_budgets) // 2]
            runner.exp_scaling(needles, preset.scaling_budgets, preset.timing_repeats, mid)
    except KeyboardInterrupt:
        print("\n[interrupted] partial results are saved; re-run the same command to resume.")
    except Exception:
        traceback.print_exc()
        print("[error] partial results are saved; re-run the same command to resume.")
    finally:
        if "report" in exps or len(exps) > 1:
            try:
                build_report(out, g.name, out / "calibration" / args.calib_source, meta)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
