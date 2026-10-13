"""
Task metrics and analytic FLOP accounting.

Each task has ONE primary quality metric (v7 averaged F1 with EM and mixed metrics):
  idbench   accuracy (answer contained)          squad      token F1 (SQuAD official normalisation)
  mmlu      accuracy                             gsm8k      accuracy (#### answer, else last number)
  humaneval pass@1 (prompt + completion run)     longbench  qa-F1 (LongBench official metric)
  needle    accuracy (code contained)
"""

import os
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List


def _norm(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, golds: List[str]) -> float:
    p = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        pr, rc = same / len(p), same / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def exact_match(pred: str, golds: List[str]) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))


def contains_answer(pred: str, ref: str) -> float:
    p = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\- ]+", " ", str(pred).lower())).strip()
    r = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\- ]+", " ", str(ref).lower())).strip()
    return float(p == r or re.search(r"\b" + re.escape(r) + r"\b", p) is not None)


def gsm8k_correct(pred: str, gold: str) -> float:
    txt = pred.replace(",", "")
    m = re.findall(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)", txt)
    nums = m or re.findall(r"-?\d+(?:\.\d+)?", txt)
    if not nums:
        return 0.0
    try:
        return float(abs(float(nums[-1]) - float(gold)) < 1e-4)
    except ValueError:
        return 0.0


def mmlu_correct(pred: str, gold: str) -> float:
    t = pred.strip().upper()
    m = re.search(r"ANSWER\s*(?:IS|:)?\s*\(?([ABCD])\b", t) or re.search(r"^\(?([ABCD])\b", t) \
        or re.search(r"\b([ABCD])\b", t)
    return float(m is not None and m.group(1) == gold)


def extract_code(completion: str, ref: Dict) -> str:
    prompt, entry = ref["prompt"], ref["entry_point"]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, flags=re.S)
    if not blocks and "```" in completion:
        blocks = [completion.split("```", 1)[1].split("\n", 1)[-1]]
    code = max(blocks, key=len) if blocks else completion
    if re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        header = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
        return header + "\n" + code
    return prompt + code


def humaneval_pass1(completion: str, ref: Dict, timeout: float = 10.0) -> float:
    program = extract_code(completion, ref) + "\n\n" + ref["test"] + f"\ncheck({ref['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return float(r.returncode == 0)
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def sst2_correct(pred: str, gold: str) -> float:
    t = pred.strip().lower()
    has_pos = bool(re.search(r"\bpositive\b", t))
    has_neg = bool(re.search(r"\bnegative\b", t))
    if has_pos and not has_neg:
        return float(gold == "positive")
    if has_neg and not has_pos:
        return float(gold == "negative")
    if t.startswith("pos"):
        return float(gold == "positive")
    if t.startswith("neg"):
        return float(gold == "negative")
    return 0.0


def score(task: str, pred: str, ref: Any) -> Dict[str, float]:
    """Returns {'quality': primary, ...secondary}."""
    if task == "idbench" or task == "needle":
        return {"quality": contains_answer(pred, ref)}
    if task == "squad":
        return {"quality": token_f1(pred, ref), "em": exact_match(pred, ref)}
    if task == "mmlu":
        return {"quality": mmlu_correct(pred, ref)}
    if task == "gsm8k":
        return {"quality": gsm8k_correct(pred, ref)}
    if task == "humaneval":
        return {"quality": humaneval_pass1(pred, ref)}
    if task == "longbench":
        return {"quality": token_f1(pred, ref), "em": exact_match(pred, ref)}
    if task == "sst2" or task == "sst":
        return {"quality": sst2_correct(pred, ref)}
    raise ValueError(task)


def rouge_l(a: str, b: str) -> float:
    x, y = a.split(), b.split()
    if not x or not y:
        return float(x == y)
    x, y = x[:400], y[:400]
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    l = prev[-1]
    if l == 0:
        return 0.0
    p, r = l / len(x), l / len(y)
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------------------- FLOPs


def prefill_flops(geom, n: int, n_halt: int = 0, attn_pairs_after: int = None, policy: str = None,
                  mlp_hidden: int = 512) -> float:
    """
    Analytic prefill FLOPs (2 x MACs) of a decoder-only model with causal attention.
    Layers < d process all n tokens; layers >= d process n - n_halt active tokens, attend
    `attn_pairs_after` (query, key) pairs, and pay the transport cost for halted tokens.
    LM head excluded (identical: only the last position is projected).
    """
    D, H, KH, hd, I = geom.hidden, geom.heads, geom.kv_heads, geom.head_dim, geom.intermediate
    Lyr, d = geom.num_layers, geom.decision_layer
    lin = 2 * D * H * hd * 2 + 2 * D * KH * hd * 2 + 2 * 3 * D * I   # q,o + k,v + gated MLP
    att = 2 * 2 * H * hd                                             # QK^T + PV per pair
    dense_pairs = n * (n + 1) // 2
    dense_layer = n * lin + dense_pairs * att
    if n_halt <= 0 or policy is None:
        return float(Lyr * dense_layer)
    n_a = n - n_halt
    pairs = attn_pairs_after if attn_pairs_after is not None else dense_pairs
    transport = 0
    if policy in ("reproject", "ridge", "ridge_layer", "ridge_plain", "ridge_gated", "mlp", "depthweave"):
        transport = n_halt * 2 * D * 2 * KH * hd                      # k,v projections of frozen state
        if policy != "reproject":
            transport *= 2                                            # + linear correction
        if policy == "mlp":
            transport += n_halt * 2 * (D * mlp_hidden + mlp_hidden * 2 * KH * hd)
    sparse_layer = n_a * lin + pairs * att + transport
    return float(d * dense_layer + (Lyr - d) * sparse_layer)
