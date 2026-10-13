#!/usr/bin/env bash
# ==============================================================================
# Pipeline:
#   Phase 1: TokenSparse downstream evaluation (SST-2 872 + SQuAD 800 + MMLU 800) @ 0.3, 0.5, 0.7, 0.9
#   Phase 2: DepthWeave downstream evaluation (SST-2 872 + SQuAD 800 + MMLU 800) @ 0.3, 0.5, 0.7, 0.9
#   Phase 3: DepthWeave LongBench full evaluation (all 3,218 prompts) @ 0.3, 0.5, 0.7, 0.9
# All runs execute sequentially in a detached background tmux session.
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SESSION_NAME="qira_eval_pipeline"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p results_tasks
mkdir -p results

MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_SAFE="Qwen2.5-7B-Instruct"
DENSE_PATH="results/$MODEL_SAFE/dense_references.pt"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "=========================================================="
    echo "Warning: tmux session '$SESSION_NAME' is already running!"
    echo "To monitor live: tail -f logs/pipeline.log"
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

BUDGETS="0.3 0.5 0.7 0.9"

echo "=========================================================="
echo "Launching Sequential Pipeline for TokenSparse & DepthWeave"
echo "Phase 1: TokenSparse Downstream (sst2, squad, mmlu)"
echo "Phase 2: DepthWeave Downstream  (sst2, squad, mmlu)"
echo "Phase 3: DepthWeave LongBench   (all 16 tasks, 3,218 samples)"
echo "Budgets: $BUDGETS"
echo "Model:   $MODEL"
echo "Python:  $PYTHON_BIN"
echo "Log:     $LOG_DIR/pipeline.log"
echo "=========================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'cd \"$PROJECT_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && {
        echo \"=== [PHASE 1/3] TokenSparse Downstream Run Starting ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods TokenSparse --tasks sst2 squad mmlu --budgets $BUDGETS --experiments main --output-dir results_tasks &&
        echo \"=== [PHASE 1/3 COMPLETE] TokenSparse Downstream Finished ===\" &&

        echo \"=== [PHASE 2/3] DepthWeave Downstream Run Starting ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods DepthWeave --tasks sst2 squad mmlu --budgets $BUDGETS --experiments main --output-dir results_tasks &&
        echo \"=== [PHASE 2/3 COMPLETE] DepthWeave Downstream Finished ===\" &&

        echo \"=== [SYNC CALIBRATION] Copying DepthWeave calibration to results/ ===\" &&
        mkdir -p results/$MODEL_SAFE &&
        if [ -d \"results_tasks/$MODEL_SAFE/calibration\" ]; then
            cp -r results_tasks/$MODEL_SAFE/calibration results/$MODEL_SAFE/
        fi &&

        echo \"=== [PHASE 3/3] DepthWeave LongBench Full Run Starting ===\" &&
        LOAD_DENSE_FLAG=\"\"
        if [ -f \"$DENSE_PATH\" ]; then
            echo \"Found cached dense references: $DENSE_PATH (skipping dense re-eval)\"
            LOAD_DENSE_FLAG=\"--load-dense $DENSE_PATH\"
        fi &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods DepthWeave --tasks longbench --longbench-all --budgets $BUDGETS \$LOAD_DENSE_FLAG --experiments main --output-dir results &&
        echo \"=== [ALL PHASES COMPLETE] Building Final Reports ===\" &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --experiments report --output-dir results_tasks &&
        $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --experiments report --output-dir results &&
        echo \"=== PIPELINE SUCCEEDED! ===\";
    } 2>&1 | tee \"$LOG_DIR/pipeline.log\"; EXIT_CODE=\$?; echo \"[Pipeline exit code \$EXIT_CODE]\"; exec bash'"

echo ""
echo "=========================================================="
echo "Pipeline successfully launched in tmux session '$SESSION_NAME'!"
echo ""
echo "Helpful Commands:"
echo "  1. View live logs: tail -f logs/pipeline.log"
echo "  2. Attach tmux:    tmux attach -t $SESSION_NAME"
echo "  3. Detach tmux:    Press Ctrl+B, then D"
echo "=========================================================="
