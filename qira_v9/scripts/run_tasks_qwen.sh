#!/usr/bin/env bash
# ==============================================================================
# Run Downstream Tasks on Qwen2.5-7B-Instruct:
# Tasks: SST-2 (872 full - resumes), SQuAD v1.1 (800), MMLU (800)
# Methods: QIRA, QIRA-L, TokenSparse, LazyLLM, SlimInfer, QuickSilver, KVSimilarity + Dense
# Expected runtime: ~2 hours 45 mins.
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SESSION_NAME="qira_tasks_qwen"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p results_tasks

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT_DIR="results_tasks"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "=========================================================="
    echo "Warning: tmux session '$SESSION_NAME' is already running!"
    echo "To monitor live: tail -f logs/tasks_qwen.log"
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

METHODS="QIRA QIRA-L TokenSparse LazyLLM SlimInfer QuickSilver KVSimilarity"
TASKS="sst2 squad mmlu"
BUDGETS="0.3 0.5 0.7 0.9"

echo "=========================================================="
echo "Launching Downstream Tasks Benchmark ($TASKS)"
echo "Task Order:   $TASKS (SST-2 resumes/skips, then SQuAD, MMLU)"
echo "Budgets:      $BUDGETS"
echo "Methods:      $METHODS + Dense"
echo "Model:        $MODEL"
echo "Project Root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Log File:     $LOG_DIR/tasks_qwen.log"
echo "Estimated:    ~2 hours 45 mins (SQuAD ~1.8h + MMLU ~1h)"
echo "=========================================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'cd \"$PROJECT_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && $PYTHON_BIN run_benchmark.py --model \"$MODEL\" --preset paper --methods $METHODS --tasks $TASKS --budgets $BUDGETS --experiments main --output-dir \"$OUT_DIR\" 2>&1 | tee \"$LOG_DIR/tasks_qwen.log\"; EXIT_CODE=\$?; echo \"[Tasks run finished with exit code \$EXIT_CODE]\"; exec bash'"

echo ""
echo "=========================================================="
echo "Downstream Tasks Benchmark successfully launched in background tmux session!"
echo "You can safely disconnect your SSH connection or close your laptop anytime."
echo ""
echo "Helpful Commands:"
echo "  1. View live logs: tail -f logs/tasks_qwen.log"
echo "  2. Attach tmux:    tmux attach -t $SESSION_NAME"
echo "  3. Detach tmux:    Press Ctrl+B, then D"
echo "=========================================================="
