#!/bin/bash
# Full pipeline for sample-alt-voters experiment (11 new topics)
# This script chains all steps and runs them sequentially

set -e  # Exit on error

cd /home/ec2-user/single-winner-generative-social-choice
export $(cat .env | xargs)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Starting full pipeline at $(date)"
echo "=============================================="

# Step 1: Wait for statement generation to complete (if running)
echo ""
echo "[Step 1] Checking if statement generation is still running..."
STMT_GEN_LOG=$(ls -t logs/generate_statements_*.log 2>/dev/null | head -1)
if [ -n "$STMT_GEN_LOG" ]; then
    echo "Found log: $STMT_GEN_LOG"
    # Wait for the process to finish by checking if tmux session exists
    while tmux has-session -t stmt_gen 2>/dev/null; do
        LAST_LINE=$(tail -1 "$STMT_GEN_LOG" 2>/dev/null || echo "")
        echo "$(date +%H:%M:%S) - Waiting for statement generation... $LAST_LINE"
        sleep 60
    done
    echo "Statement generation complete!"
fi

# Verify all 13 topics have Alt1 and Alt4 files
echo ""
echo "[Step 1b] Verifying statement files..."
ALT1_COUNT=$(ls data/sample-alt-voters/sampled-statements/persona_no_context/*.json 2>/dev/null | wc -l)
ALT4_COUNT=$(ls data/sample-alt-voters/sampled-statements/no_persona_no_context/*.json 2>/dev/null | wc -l)
echo "Alt1 files: $ALT1_COUNT, Alt4 files: $ALT4_COUNT"

if [ "$ALT1_COUNT" -lt 13 ] || [ "$ALT4_COUNT" -lt 13 ]; then
    echo "WARNING: Not all statement files present. Running statement generation..."
    uv run python -m src.sample_alt_voters.generate_statements --all 2>&1 | tee "$LOG_DIR/generate_statements_${TIMESTAMP}.log"
fi

echo "Alt1 files:"
ls data/sample-alt-voters/sampled-statements/persona_no_context/
echo "Alt4 files:"
ls data/sample-alt-voters/sampled-statements/no_persona_no_context/

# Step 2: Generate per-rep statements (Alt2 + Alt3)
echo ""
echo "=============================================="
echo "[Step 2] Generating per-rep statements (Alt2 + Alt3)..."
echo "Started at $(date)"
echo "=============================================="
uv run python -m src.sample_alt_voters.generate_per_rep_statements --all 2>&1 | tee "$LOG_DIR/generate_per_rep_${TIMESTAMP}.log"
echo "Per-rep generation complete at $(date)"

# Step 3: Run main experiment - Uniform voter distribution
echo ""
echo "=============================================="
echo "[Step 3] Running experiment (uniform voter distribution)..."
echo "Started at $(date)"
echo "=============================================="
uv run python -m src.sample_alt_voters.run_experiment \
    --voter-dist uniform --all-topics --all-alts 2>&1 | tee "$LOG_DIR/experiment_uniform_${TIMESTAMP}.log"
echo "Uniform experiment complete at $(date)"

# Step 4: Run main experiment - Clustered voter distribution
echo ""
echo "=============================================="
echo "[Step 4] Running experiment (clustered voter distribution)..."
echo "Started at $(date)"
echo "=============================================="
uv run python -m src.sample_alt_voters.run_experiment \
    --voter-dist clustered --all-topics --all-alts 2>&1 | tee "$LOG_DIR/experiment_clustered_${TIMESTAMP}.log"
echo "Clustered experiment complete at $(date)"

# Step 5: Run GPT*** (triple star)
echo ""
echo "=============================================="
echo "[Step 5] Running GPT*** (triple star)..."
echo "Started at $(date)"
echo "=============================================="
uv run python -m src.sample_alt_voters.run_triple_star --all 2>&1 | tee "$LOG_DIR/triple_star_${TIMESTAMP}.log"
echo "Triple star complete at $(date)"

# Step 6: Generate visualizations
echo ""
echo "=============================================="
echo "[Step 6] Generating visualizations..."
echo "Started at $(date)"
echo "=============================================="
uv run python -m src.sample_alt_voters.visualizer --all 2>&1 | tee "$LOG_DIR/visualizer_${TIMESTAMP}.log"
echo "Visualization complete at $(date)"

# Done!
echo ""
echo "=============================================="
echo "FULL PIPELINE COMPLETE!"
echo "Finished at $(date)"
echo "=============================================="
echo ""
echo "Results in: outputs/sample_alt_voters/"
echo "Logs in: $LOG_DIR/"
