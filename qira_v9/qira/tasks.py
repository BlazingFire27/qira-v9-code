"""
Evaluation tasks, calibration pools and prompt formatting.

Fixes relative to the v7 benchmark:
  * prompts are wrapped with the model's chat template (instruct models) unless disabled;
  * over-long prompts are truncated in the MIDDLE (LongBench protocol) so the question and
    the generation prompt are always preserved (v7 truncated from the right, which deleted
    the question of every long LongBench document);
  * SQuAD v1.1 (always answerable) replaces SQuAD v2 with an unanswerable-scoring bug;
  * LongBench uses the official per-dataset instructions and qa-F1 metric, 4 QA subsets;
  * calibration pools (natural / code / IDBench) are disjoint from every evaluation subset.
"""

import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qira.config import IDBENCH_REDUNDANCY, IDBENCH_SIGNAL_UNITS

# ----------------------------------------------------------------------------- formatting


class PromptFormatter:
    def __init__(self, tokenizer, use_chat: bool, max_context: int):
        self.tok = tokenizer
        self.use_chat = bool(use_chat and getattr(tokenizer, "chat_template", None))
        self.max_context = int(max_context)

    def text(self, ex: Dict) -> str:
        if self.use_chat:
            msgs = [{"role": "user", "content": ex["user"]}]
            # enable_thinking=False disables Qwen3 "thinking" mode; other templates ignore the variable
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        return ex["user"]

    def encode(self, ex: Dict, max_context: Optional[int] = None) -> List[int]:
        limit = int(max_context or self.max_context)
        ids = self.tok(self.text(ex), add_special_tokens=not self.use_chat)["input_ids"]
        if len(ids) > limit:
            half = limit // 2
            ids = ids[:half] + ids[len(ids) - (limit - half):]
        return ids

    def answer_ids(self, ex: Dict) -> List[int]:
        ans = ex.get("answer_text")
        if not ans:
            return []
        txt = ans if self.use_chat else " " + ans
        ids = self.tok(txt, add_special_tokens=False)["input_ids"]
        return ids[:48]


# ----------------------------------------------------------------------------- HF datasets


def _load_hf(name: str, config: Optional[str], split: str):
    from datasets import load_dataset
    try:
        return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    except Exception as e:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(name, repo_type="dataset", revision="refs/convert/parquet")
        files = [f for f in files if f.endswith(".parquet") and f"/{split}/" in f]
        if config:
            scoped = [f for f in files if f.startswith(f"{config}/")]
            files = scoped or files
        if not files:
            raise RuntimeError(f"cannot load {name}/{config}:{split}: {e}")
        urls = [f"hf://datasets/{name}@refs%2Fconvert%2Fparquet/{f}" for f in files]
        return load_dataset("parquet", data_files={split: urls}, split=split)


def load_squad(n: int) -> List[Dict]:
    ds = _load_hf("rajpurkar/squad", None, "validation").shuffle(seed=1234)
    out = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        answers = list(dict.fromkeys(ex["answers"]["text"]))
        out.append(dict(
            id=f"squad_{i}", task="squad", reference=answers, answer_text=answers[0],
            user=("Answer the question using only the given context. Reply with a short answer "
                  "(a few words copied from the context), nothing else.\n\n"
                  f"Context: {ex['context']}\n\nQuestion: {ex['question']}\nAnswer:"),
            meta={}))
    return out


def load_gsm8k(n: int) -> List[Dict]:
    ds = _load_hf("openai/gsm8k", "main", "test").shuffle(seed=1234)
    out = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(
            id=f"gsm8k_{i}", task="gsm8k", reference=gold, answer_text=None,
            user=("Solve the following math problem step by step. At the end, write the final "
                  f"numeric answer on its own line as '#### <number>'.\n\nProblem: {ex['question']}\nSolution:"),
            meta={}))
    return out


def load_mmlu(n: int) -> List[Dict]:
    ds = _load_hf("cais/mmlu", "all", "test").shuffle(seed=1234)
    L = ["A", "B", "C", "D"]
    out = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        opts = "\n".join(f"{L[j]}. {c}" for j, c in enumerate(ex["choices"]))
        out.append(dict(
            id=f"mmlu_{i}", task="mmlu", reference=L[ex["answer"]], answer_text=L[ex["answer"]],
            user=("Answer the following multiple-choice question with a single letter (A, B, C, or D).\n\n"
                  f"Question: {ex['question']}\n{opts}\nAnswer:"),
            meta={"subject": ex.get("subject", "")}))
    return out


def load_humaneval(n: int, use_chat: bool) -> List[Dict]:
    ds = _load_hf("openai/openai_humaneval", None, "test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        if use_chat:
            user = ("Complete the following Python function. Reply with the complete function "
                    "(including the signature) in a single ```python code block.\n\n```python\n"
                    f"{ex['prompt']}```")
        else:
            user = ex["prompt"]
        out.append(dict(
            id=ex["task_id"], task="humaneval", answer_text=None, user=user,
            reference={"prompt": ex["prompt"], "test": ex["test"], "entry_point": ex["entry_point"]},
            meta={}))
    return out


def load_sst2(n: int) -> List[Dict]:
    ds = _load_hf("stanfordnlp/sst2", None, "validation")
    out = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        ref = "positive" if ex["label"] == 1 else "negative"
        user = ("Determine the sentiment of the following movie review. "
                "Reply with only a single word: 'positive' or 'negative'.\n\n"
                f"Review: {ex['sentence']}\nAnswer:")
        out.append(dict(
            id=f"sst2_{i}", task="sst2", reference=ref, answer_text=ref,
            user=user, meta={"label": ex["label"]}))
    return out


# ----------------------------------------------------------------------------- LongBench

LB_PROMPTS = {
    "narrativeqa": ("You are given a story, which can be either a novel or a movie script, and a question. "
                    "Answer the question as concisely as you can, using a single phrase if possible. Do not "
                    "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story "
                    "as concisely as you can, using a single phrase if possible. Do not provide any explanation."
                    "\n\nQuestion: {input}\n\nAnswer:"),
    "qasper": ("You are given a scientific article and a question. Answer the question as concisely as you can, "
               "using a single phrase or sentence if possible. If the question cannot be answered based on the "
               "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
               "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n "
               "Answer the question based on the above article as concisely as you can, using a single phrase or "
               "sentence if possible. If the question cannot be answered based on the information in the article, "
               "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
               "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"),
    "multifieldqa_en": ("Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
                        "question based on the above text, only give me the answer and do not output any other "
                        "words.\n\nQuestion: {input}\nAnswer:"),
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
                 "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"),
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the report.\n\n"
                   "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"),
    "qmsum": ("You are given several passages of meetings. Write a summary of the meetings.\n\n"
              "Meetings:\n{context}\n\nNow, write a summary of the meetings.\n\nSummary:"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all news. \n\nNews:\n"
                   "{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"),
    "samsum": ("Summarize the dialogue briefly.\n\nDialogue:\n{context}\n\nSummary:"),
    "trec": ("Please determine the category of the question from the context.\n\n{context}\n\nQuestion: {input}\nAnswer:"),
    "triviaqa": ("Answer the question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nThe following are given passages.\n{context}\n\nQuestion: {input}\nAnswer:"),
    "passage_count": ("There are some paragraphs below. Please count how many distinct paragraphs there are.\n\n"
                      "{context}\n\nQuestion: {input}\nAnswer:"),
    "passage_retrieval_en": ("Here are 30 paragraphs from Wikipedia. Please find the paragraph that contains the key "
                             "to the question.\n\n{context}\n\nQuestion: {input}\nAnswer:"),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
LB_PROMPTS["musique"] = LB_PROMPTS["hotpotqa"]
LB_PROMPTS["2wikimqa"] = LB_PROMPTS["hotpotqa"]


class LongBench:
    URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rows: Dict[str, List[Dict]] = {}

    def _ensure(self):
        zip_path = self.cache_dir / "data.zip"
        if not list(self.cache_dir.rglob("narrativeqa.jsonl")):
            if not zip_path.exists():
                print("[LongBench] downloading official data.zip ...")
                urllib.request.urlretrieve(self.URL, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.cache_dir)

    def rows(self, subset: str) -> List[Dict]:
        if subset not in self._rows:
            self._ensure()
            matches = list(self.cache_dir.rglob(f"{subset}.jsonl"))
            if not matches:
                raise FileNotFoundError(f"LongBench subset {subset} not found")
            with open(matches[0], encoding="utf-8") as f:
                self._rows[subset] = [json.loads(l) for l in f if l.strip()]
        return self._rows[subset]

    def examples(self, subset: str, n: int, seed: int, prefix: str) -> List[Dict]:
        rows = list(self.rows(subset))
        random.Random(seed).shuffle(rows)
        out = []
        for i, ex in enumerate(rows[:n]):
            ans = ex.get("answers") or [""]
            if isinstance(ans, str):
                ans = [ans]
            user = LB_PROMPTS[subset].format(context=ex.get("context", ""), input=ex.get("input", ""))
            out.append(dict(id=f"{prefix}_{subset}_{i}", task="longbench", reference=ans,
                            answer_text=ans[0] if ans and len(ans[0]) < 200 else None, user=user,
                            meta={"subset": subset, "length": ex.get("length")}))
        return out


def load_longbench_eval(lb: LongBench, n: int, subsets: List[str]) -> List[Dict]:
    per = max(1, n // len(subsets))
    out = []
    for j, s in enumerate(subsets):
        out += lb.examples(s, per, seed=20260922 + j, prefix="lb")
    return out


# ----------------------------------------------------------------------------- IDBench (frozen)

ID_FACTS = [
    ("capital", "What is the capital of France?", "Paris", [
        "France's capital city is Paris.", "Paris is the capital of France.",
        "The national capital of France is Paris.", "Paris serves as France's capital.",
        "France has its capital in Paris.", "Paris is France's seat of government.",
        "The French capital is Paris.", "Paris is the principal city and capital of France.",
        "The capital city named for France in this question is Paris.", "France's capital is the city of Paris.",
        "Paris is the answer to the capital-city question.", "The relevant capital-city fact is that Paris is France's capital."]),
    ("planet", "Which planet is the largest in the Solar System?", "Jupiter", [
        "Jupiter is the largest planet in the Solar System.", "The Solar System's largest planet is Jupiter.",
        "Jupiter has the greatest planetary size in our Solar System.", "The largest Solar System planet is Jupiter.",
        "Jupiter is the largest of the planets orbiting the Sun.", "Among the Solar System planets, Jupiter is largest.",
        "Jupiter is the answer to the largest-planet question.", "The largest planet around the Sun is Jupiter.",
        "Jupiter has the greatest diameter of the Solar System planets.", "The Solar System's biggest planet is Jupiter.",
        "Jupiter is the relevant largest-planet fact.", "The largest planetary body among the eight planets is Jupiter."]),
    ("gold", "What is the chemical symbol for gold?", "Au", [
        "Gold has the chemical symbol Au.", "The symbol used for gold is Au.",
        "Au is the chemical symbol of gold.", "Gold is represented chemically by Au.",
        "The element gold is written as Au in chemical notation.", "Au denotes the element gold.",
        "The requested gold symbol is Au.", "Gold's standard chemical abbreviation is Au.",
        "In the periodic table, gold is Au.", "Au is the symbol assigned to gold.",
        "The relevant element-symbol fact is gold equals Au.", "Gold is identified by the symbol Au."]),
    ("water", "What is the boiling point of water at sea level in Celsius?", "100", [
        "At sea level, water boils at 100 degrees Celsius.", "Water's sea-level boiling point is 100 C.",
        "The standard boiling point of water at sea level is 100 degrees Celsius.",
        "At one atmosphere, water boils at 100 Celsius.", "Water reaches its normal boiling point at 100 C.",
        "The Celsius boiling point of water at sea level is 100.", "100 degrees Celsius is water's normal boiling point.",
        "At sea-level pressure, water boils at 100 degrees Celsius.", "The relevant temperature is 100 C.",
        "Water's boiling temperature under standard sea-level conditions is 100 C.",
        "The answer to the boiling-point question is 100 degrees Celsius.", "At standard atmospheric pressure, water boils at 100 C."]),
]
DISTRACTORS = [
    "The Pacific Ocean is the largest ocean on Earth.", "A triangle has three sides.",
    "The Moon orbits Earth.", "Mount Everest is in the Himalayas.",
    "A byte commonly contains eight bits.", "The Earth rotates on its axis.",
    "The Amazon rainforest is in South America.", "Mercury is the closest planet to the Sun.",
    "The Nile is a major river in Africa.", "Photosynthesis occurs in plants and algae.",
    "The Pacific is larger than the Atlantic Ocean.", "A leap year normally has an extra day in February.",
    "The human heart pumps blood through the circulatory system.", "Sound travels through a medium.",
    "Earth has one natural satellite, the Moon.", "Carbon dioxide contains carbon and oxygen.",
    "The Sahara is a large desert in Africa.", "The Sun is a star.",
    "Water freezes at 0 degrees Celsius under standard pressure.", "A square has four equal sides.",
]


def _idbench_example(fact, redundancy, idx, rng, split):
    key, question, answer, signals = fact
    total = IDBENCH_SIGNAL_UNITS if redundancy == 0 else int(round(IDBENCH_SIGNAL_UNITS / (1.0 - redundancy / 100.0)))
    nd = total - IDBENCH_SIGNAL_UNITS
    units = list(signals) + [DISTRACTORS[(idx * 17 + j) % len(DISTRACTORS)] for j in range(max(0, nd))]
    rng.shuffle(units)
    context = "\n".join(f"{j + 1}. {u}" for j, u in enumerate(units))
    user = ("Answer the question using the information in the context. Ignore unrelated statements. "
            "Give only the answer.\n\nContext:\n" + context + f"\n\nQuestion: {question}\nAnswer:")
    return dict(id=f"idb_{split}_{key}_{redundancy}_{idx}", task="idbench", reference=answer,
                answer_text=answer, user=user,
                meta={"redundancy_pct": redundancy, "fact_group": key, "split": split})


def build_idbench(cal_per_density: int = 4, eval_per_density: int = 6) -> Tuple[List[Dict], List[Dict]]:
    """Identical generator/seed to the v7 benchmark; calibration items are always the first 4 per density."""
    rng = random.Random(20260922)
    cal, ev = [], []
    for density in IDBENCH_REDUNDANCY:
        for i in range(4 + max(eval_per_density, 6)):
            fact = ID_FACTS[i % len(ID_FACTS)]
            split = "calibration" if i < 4 else "evaluation"
            ex = _idbench_example(fact, density, i, rng, split)
            if split == "calibration" and i < cal_per_density:
                cal.append(ex)
            elif split == "evaluation" and (i - 4) < eval_per_density:
                ev.append(ex)
    return cal, ev


# ----------------------------------------------------------------------------- needle (context scaling)


def build_needle_examples(lb: LongBench, tokenizer, lengths: List[int], per_length: int,
                          formatter: PromptFormatter) -> List[Dict]:
    docs = [r["context"] for r in lb.rows("gov_report")]
    random.Random(7).shuffle(docs)
    hay = "\n\n".join(docs[:200])
    hay_ids = tokenizer(hay, add_special_tokens=False)["input_ids"]
    rng = random.Random(4242)
    depths = [0.25, 0.5, 0.75, 0.1, 0.9]
    out = []
    for T in lengths:
        for j in range(per_length):
            code = str(rng.randint(1000000, 9999999))
            needle = f" The secret access code is {code}. Remember this code. "
            question = "\n\nWhat is the secret access code mentioned in the document above? Reply with the number only."
            overhead = len(tokenizer(needle + question + "Read the following document.\n\n",
                                     add_special_tokens=False)["input_ids"]) + 64
            body = max(256, T - overhead)
            body_ids = hay_ids[j * 997: j * 997 + body]
            text = tokenizer.decode(body_ids)
            depth = depths[j % len(depths)]
            cut = int(len(text) * depth)
            nxt = text.find(". ", cut)
            cut = nxt + 2 if 0 <= nxt < cut + 400 else cut
            doc = text[:cut] + needle + text[cut:]
            out.append(dict(id=f"needle_{T}_{j}", task="needle", reference=code, answer_text=code,
                            user="Read the following document.\n\n" + doc + question,
                            meta={"target_len": T, "depth": depth}))
    return out


# ----------------------------------------------------------------------------- calibration pools


def build_calibration_pool(source: str, lb: LongBench, n: int, idbench_cal: List[Dict]) -> List[Dict]:
    from qira.config import LONGBENCH_CALIB_CODE, LONGBENCH_CALIB_NATURAL
    if source == "idbench":
        return list(idbench_cal)[:n] if n else list(idbench_cal)
    subsets = LONGBENCH_CALIB_NATURAL if source == "natural" else LONGBENCH_CALIB_CODE if source == "code" else None
    if subsets is None:
        raise ValueError(f"unknown calibration source {source}")
    per = max(1, n // len(subsets))
    out = []
    for j, s in enumerate(subsets):
        out += lb.examples(s, per, seed=777 + j, prefix=f"cal_{source}")
    return out
