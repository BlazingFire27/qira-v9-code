#!/usr/bin/env bash
# ==============================================================================
# Phase 3: Merge Shards and Generate Final Paper Tables & Figures
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Detect python binary
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

MODEL_SAFE="Qwen2.5-7B-Instruct"
FINAL_DIR="results/$MODEL_SAFE"
mkdir -p "$FINAL_DIR"

echo "=========================================================="
echo "Phase 3: Merging Results from All 3 GPUs"
echo "Target Directory: $FINAL_DIR"
echo "Python:           $PYTHON_BIN"
echo "Time:             $(date)"
echo "=========================================================="

# 1. Merge main experiment results (avoid duplicate Dense rows by sorting -u on _key)
$PYTHON_BIN -c '
import json
from pathlib import Path

final_dir = Path("'$FINAL_DIR'")
rows = {}

# 1. Load Dense reference
dense_file = Path("results/'$MODEL_SAFE'/Dense.jsonl")
if dense_file.exists():
    for line in dense_file.open("r", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["_key"]] = r

# 2. Load GPU shards
for g in ["results_gpu1", "results_gpu2", "results_gpu3"]:
    main_p = Path(g) / "'$MODEL_SAFE'" / "main.jsonl"
    if main_p.exists():
        print(f"Loading {main_p} ...")
        for line in main_p.open("r", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["_key"]] = r

out_main = final_dir / "main.jsonl"
with out_main.open("w", encoding="utf-8") as f:
    for r in rows.values():
        f.write(json.dumps(r) + "\n")
print(f"Merged {len(rows)} unique evaluation rows into {out_main}")

# 3. Merge latency experiment results if present
lat_rows = {}
for g in ["results_gpu1", "results_gpu2", "results_gpu3"]:
    lat_p = Path(g) / "'$MODEL_SAFE'" / "latency.jsonl"
    if lat_p.exists():
        print(f"Loading {lat_p} ...")
        for line in lat_p.open("r", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                lat_rows[r["_key"]] = r

if lat_rows:
    out_lat = final_dir / "latency.jsonl"
    with out_lat.open("w", encoding="utf-8") as f:
        for r in lat_rows.values():
            f.write(json.dumps(r) + "\n")
    print(f"Merged {len(lat_rows)} latency rows into {out_lat}")
'

# Copy calibration and config so build_report produces Fig 1 and all NMSE/alignment tables
if [ -d "results_gpu1/$MODEL_SAFE/calibration" ] && [ ! -d "$FINAL_DIR/calibration" ]; then
    cp -r "results_gpu1/$MODEL_SAFE/calibration" "$FINAL_DIR/"
fi
if [ -f "results_gpu1/$MODEL_SAFE/run_config.json" ] && [ ! -f "$FINAL_DIR/run_config.json" ]; then
    cp "results_gpu1/$MODEL_SAFE/run_config.json" "$FINAL_DIR/"
fi

# 2. Run the reporting pipeline to produce tables and figures
echo "Generating Tables and Figures via qira.report ..."
$PYTHON_BIN run_benchmark.py --experiments report --model "Qwen/Qwen2.5-7B-Instruct"

echo "=========================================================="
echo "Phase 3 COMPLETE! All tables and figures generated in:"
echo "  $FINAL_DIR/tables/"
echo "  $FINAL_DIR/figures/"
echo "Time: $(date)"
echo "=========================================================="
