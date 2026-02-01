#!/bin/bash
# =============================================================================
# Run Pipeline for Selected Topics
# =============================================================================
# 
# This script runs the full sample-alt-voters experiment pipeline for a subset
# of 6 topics instead of all 13.
#
# Selected topics:
#   - policing: Policing strategies
#   - trust: Public trust in institutions
#   - environment: Environment vs. economy
#   - abortion: Abortion laws
#   - electoral: Electoral College reform
#   - healthcare: Healthcare access
#
# Usage:
#   # Run in foreground with logging
#   ./scripts/run_6topics.sh
#
#   # Run in tmux (recommended for long runs)
#   tmux new -s six_topics './scripts/run_6topics.sh'
#
# Expected runtime: 10-20+ hours depending on API rate limits
# Expected cost: ~$1,434 (6 topics × ~$239/topic)
# =============================================================================

set -e

# Configuration
TOPICS="policing trust environment abortion electoral healthcare"
LOG_FILE="logs/pipeline_6topics_$(date +%Y%m%d_%H%M%S).log"

# Ensure we're in the project root
cd "$(dirname "$0")/.."
mkdir -p logs

echo "Starting 6-topic pipeline at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Topics: $TOPICS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# =============================================================================
# Stage 1: Run experiments for all 6 topics (uniform + clustered)
# =============================================================================
for topic in $TOPICS; do
  echo "" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
  echo "=== Running $topic (uniform) ===" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
  uv run python -m src.sample_alt_voters.run_experiment \
    --voter-dist uniform --topic "$topic" --all-alts 2>&1 | tee -a "$LOG_FILE"
  
  echo "" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
  echo "=== Running $topic (clustered) ===" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
  uv run python -m src.sample_alt_voters.run_experiment \
    --voter-dist clustered --topic "$topic" --all-alts 2>&1 | tee -a "$LOG_FILE"
done

# =============================================================================
# Stage 2: Run GPT**, GPT***, and Random Insertion with batched iterative ranking
# =============================================================================
# These methods generate new statements and use batched iterative ranking
# to compute accurate epsilon values without position bias.
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "=== Stage 2: Run GPT**/GPT***/Random ===" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
uv run python scripts/run_gpt_star_batched.py 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Stage 3: Generate visualization plots
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "=== Stage 4: Visualize ===" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
uv run python -m src.sample_alt_voters.visualizer --all 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Done
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Pipeline completed at $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
