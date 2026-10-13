"""Append-only JSONL result store with resume (one file per experiment)."""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        try:
            return _clean(v.item())
        except Exception:
            pass
    return v


class ResultStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        self.keys = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # torn last line after a crash
                    self.rows.append(r)
                    self.keys.add(r.get("_key"))

    @staticmethod
    def make_key(*parts) -> str:
        return "|".join(str(p) for p in parts)

    def done(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str, row: Dict[str, Any]):
        row = {k: _clean(v) for k, v in row.items()}
        row["_key"] = key
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self.keys.add(key)

    def get(self, key: str):
        for r in reversed(self.rows):
            if r.get("_key") == key:
                return r
        return None

    def reset(self):
        if self.path.exists():
            self.path.unlink()
        self.rows, self.keys = [], set()
