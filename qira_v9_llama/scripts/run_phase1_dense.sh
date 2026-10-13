#!/usr/bin/env bash
# ==============================================================================
# Phase 1: Dense Reference Precomputation (Run on 1 GPU: RTX PRO 6000 S or A6000)
# Runs inside a persistent detached tmux session with full logging.
# Safe from SSH disconnects and laptop sleep.
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SESSION_NAME="qira_phase1_dense"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p results

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "=========================================================="
    echo "Warning: tmux session '$SESSION_NAME' is already running!"
    echo "To monitor live: tail -f logs/phase1_dense.log"
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
echo "Launching Phase 1 (Dense Precomputation) inside tmux: $SESSION_NAME"
echo "Project Root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Log File:     $LOG_DIR/phase1_dense.log"
echo "=========================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'cd \"$PROJECT_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && $PYTHON_BIN run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --dense-only --longbench-all --tasks longbench --output-dir results 2>&1 | tee \"$LOG_DIR/phase1_dense.log\"; echo \"[Process finished with exit code \$?]\"; exec bash'"

echo ""
echo "=========================================================="
echo "Phase 1 successfully launched in background tmux session!"
echo "You can safely disconnect your SSH connection anytime."
echo ""
echo "Helpful Commands:"
echo "  1. View live logs: tail -f logs/phase1_dense.log"
echo "  2. Attach tmux:    tmux attach -t $SESSION_NAME"
echo "  3. Detach tmux:    Press Ctrl+B, then D"
echo "=========================================================="
