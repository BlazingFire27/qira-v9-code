#!/usr/bin/env bash
# ==============================================================================
# Turnkey One-Shot Automated Pipeline for LLaMA-3.1-8B-Instruct
# Evaluates:
#   1. Offline Calibration (Ridge + Layer-Ridge + DepthWeave)
#   2. Dense Reference Precomputation (LongBench 16 tasks + SST-2 + SQuAD + MMLU)
#   3. Downstream Tasks (SST-2 full, SQuAD v1.1 800, MMLU 800) across 8 Baselines @ 0.3, 0.5, 0.7, 0.9
#   4. LongBench Full Suite (all 16 subsets, 3,218 samples) across 8 Baselines @ 0.3, 0.5, 0.7, 0.9
#   5. Context Scaling Benchmark (2K, 4K, 8K, 16K, 32K) @ 0.5, 0.7, 0.9
#   6. Automated Summary Report & CSV Generation
#   7. Packaging of all metrics & tables (excluding .pt) for single-command SCP download
#
# Execution: Entirely hands-off inside a persistent detached tmux session.
# Expected runtime: ~32-38 hours total on a single high-memory GPU (Blackwell / H100 / A100 / A6000).
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SESSION_NAME="llama_eval_pipeline"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p results_llama

# Model identifier (defaults to official meta-llama; automatically falls back to ungated NousResearch mirror if no token)
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="results_llama"
MODEL_SAFE="Llama-3.1-8B-Instruct"
DENSE_PATH="$OUT_DIR/$MODEL_SAFE/dense_references.pt"

# Baselines in strict canonical order
METHODS="QIRA QIRA-L TokenSparse DepthWeave LazyLLM SlimInfer QuickSilver KVSimilarity"
DOWNSTREAM_TASKS="sst2 squad mmlu"
BUDGETS="0.3 0.5 0.7 0.9"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "=========================================================="
    echo "Warning: tmux session '$SESSION_NAME' is already running!"
    echo "To monitor live: tail -f $LOG_DIR/master_llama_pipeline.log"
    echo "To attach:       tmux attach -t $SESSION_NAME"
    echo "To kill:         tmux kill-session -t $SESSION_NAME"
    echo "=========================================================="
    exit 0
fi

# Detect python binary
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

echo "=========================================================="
echo "LAUNCHING UNIFIED TURNKEY LLAMA-3.1-8B EVALUATION PIPELINE"
echo "Model:        $MODEL (auto-mirror fallback enabled)"
echo "Methods:      Dense + $METHODS"
echo "Budgets:      $BUDGETS"
echo "Tasks:        LongBench (All 16 subsets) + SST-2 + SQuAD + MMLU"
echo "Scaling:      2K, 4K, 8K, 16K, 32K context windows"
echo "Session:      tmux attach -t $SESSION_NAME"
echo "Master Log:   $LOG_DIR/master_llama_pipeline.log"
echo "Estimated:    ~32-38 hours hands-off execution"
echo "=========================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'set -o pipefail && cd \"$PROJECT_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && {
        START_TIME=\$(date) &&
        echo \"[PIPELINE START] \$START_TIME\" &&

        # ------------------------------------------------------------------
        # STAGE 1: Offline Cross-Layer Transport Calibration (~5-10 mins)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 1/6] Running Offline Calibration (Ridge, Layer-Ridge, DepthWeave) ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --experiments calibrate --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 1 COMPLETE] Calibration Matrix Stored in $OUT_DIR/$MODEL_SAFE/calibration/ ===\" &&

        # ------------------------------------------------------------------
        # STAGE 2: Dense Reference Precomputation (~4-5 hours)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 2/6] Precomputing Dense Ground-Truth Cache (LongBench + Downstream) ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --dense-only --longbench-all --tasks longbench $DOWNSTREAM_TASKS --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 2 COMPLETE] Dense References Cached at $DENSE_PATH ===\" &&

        # ------------------------------------------------------------------
        # STAGE 3: Core Downstream Tasks Evaluation (~4-6 hours)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 3/6] Running Downstream Tasks ($DOWNSTREAM_TASKS) across All 8 Baselines ===\" &&
        LOAD_DENSE_FLAG=\"\"
        if [ -f \"$DENSE_PATH\" ]; then
            LOAD_DENSE_FLAG=\"--load-dense $DENSE_PATH\"
        fi &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods $METHODS --tasks $DOWNSTREAM_TASKS --budgets $BUDGETS \$LOAD_DENSE_FLAG --experiments main --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 3 COMPLETE] Downstream Tasks Finished ===\" &&

        # ------------------------------------------------------------------
        # STAGE 4: LongBench Full 16-Subset Suite (~24-28 hours)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 4/6] Running Full LongBench (All 16 Subsets, 3,218 Prompts) across All 8 Baselines ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods $METHODS --tasks longbench --longbench-all --budgets $BUDGETS \$LOAD_DENSE_FLAG --experiments main --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 4 COMPLETE] Full LongBench Suite Finished ===\" &&

        # ------------------------------------------------------------------
        # STAGE 5: Long-Context Scaling Benchmark (2K, 4K, 8K, 16K, 32K) (~1-2 hours)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 5/6] Running Long-Context Scaling Benchmark (2K, 4K, 8K, 16K, 32K) ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --experiments scaling --scaling-lengths 2048 4096 8192 16384 32768 --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 5 COMPLETE] Context Scaling Benchmark Finished ===\" &&

        # ------------------------------------------------------------------
        # STAGE 6: Automated Summary Report & Table Generation (~1 min)
        # ------------------------------------------------------------------
        echo \"=== [STAGE 6/6] Generating Master Comparative Tables & CSV Exports ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --experiments report --output-dir \"$OUT_DIR\" &&
        echo \"=== [STAGE 6 COMPLETE] Master Reports Generated in $OUT_DIR/$MODEL_SAFE/tables/ ===\" &&

        # ------------------------------------------------------------------
        # STAGE 7: Package all Non-PT Artifacts for Single-Command Download
        # ------------------------------------------------------------------
        echo \"=== [PACKAGING] Creating Lightweight Metrics Archive (Excluding .pt binaries) ===\" &&
        ARCHIVE_NAME=\"results_llama_all_metrics.tar.gz\" &&
        tar --exclude=\"*.pt\" --exclude=\"*.safetensors\" --exclude=\"*.bin\" -czf \"$OUT_DIR/\$ARCHIVE_NAME\" -C \"$OUT_DIR\" \"$MODEL_SAFE\" &&
        echo \"=== Archive Created: $OUT_DIR/\$ARCHIVE_NAME ===\" &&

        END_TIME=\$(date) &&
        echo \"==========================================================\" &&
        echo \"ALL LLAMA BENCHMARKS COMPLETED SUCCESSFULLY!\" &&
        echo \"Start Time: \$START_TIME\" &&
        echo \"End Time:   \$END_TIME\" &&
        echo \"==========================================================\" &&
        echo -e \"LLaMA Full Pipeline Completed Successfully\nStart: \$START_TIME\nEnd: \$END_TIME\nArtifact: $OUT_DIR/\$ARCHIVE_NAME\" > \"$OUT_DIR/ALL_LLAMA_RUNS_COMPLETE.txt\";
    } 2>&1 | tee \"$LOG_DIR/master_llama_pipeline.log\"; EXIT_CODE=\$?; echo \"[Pipeline exit code \$EXIT_CODE]\"; exec bash'"

echo ""
echo "=========================================================="
echo "Pipeline successfully launched in tmux session '$SESSION_NAME'!"
echo ""
echo "Helpful Monitoring Commands:"
echo "  1. Live Stream: tail -f $LOG_DIR/master_llama_pipeline.log"
echo "  2. Attach tmux: tmux attach -t $SESSION_NAME"
echo "  3. Detach tmux: Press Ctrl+B, then D"
echo "  4. Check Status: grep STAGE $LOG_DIR/master_llama_pipeline.log"
echo "=========================================================="
