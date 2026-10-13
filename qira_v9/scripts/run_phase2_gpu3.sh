#!/usr/bin/env bash
# ==============================================================================
# Phase 2 (GPU 3): QuickSilver, QuickSilver+T, and KVSimilarity
# Reuses pre-computed dense_references.pt from Phase 1 (No redundant Dense passes)
# Runs inside a persistent detached tmux session with full logging.
# Safe from SSH disconnects and laptop sleep.
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SESSION_NAME="qira_phase2_gpu3"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p results_gpu3

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT_DIR="results_gpu3"
DENSE_PATH="results/Qwen2.5-7B-Instruct/dense_references.pt"

if [ ! -f "$DENSE_PATH" ]; then
    echo "ERROR: Missing $DENSE_PATH! Run Phase 1 Dense first and ensure dense_references.pt is copied here."
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "=========================================================="
    echo "Warning: tmux session '$SESSION_NAME' is already running!"
    echo "To monitor live: tail -f logs/phase2_gpu3.log"
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
echo "Launching Phase 2 (GPU 3: QuickSilver, QuickSilver+T, KVSimilarity) inside tmux: $SESSION_NAME"
echo "Project Root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Log File:     $LOG_DIR/phase2_gpu3.log"
echo "=========================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'cd \"$PROJECT_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --methods QuickSilver QuickSilver+T KVSimilarity --load-dense \"$DENSE_PATH\" --longbench-all --tasks longbench --experiments main latency --output-dir \"$OUT_DIR\" 2>&1 | tee \"$LOG_DIR/phase2_gpu3.log\"; echo \"[Process finished with exit code \$?]\"; exec bash'"

echo ""
echo "=========================================================="
echo "Phase 2 (GPU 3) successfully launched in background tmux session!"
echo "You can safely disconnect your SSH connection or close your laptop anytime."
echo ""
echo "Helpful Commands:"
echo "  1. View live logs: tail -f logs/phase2_gpu3.log"
echo "  2. Attach tmux:    tmux attach -t $SESSION_NAME"
echo "  3. Detach tmux:    Press Ctrl+B, then D"
echo "=========================================================="
