#!/bin/bash
# Run clustered pipeline: conservative first, then progressive
# Each ideology: 6 topics × 9 reps = 54 conditions

cd /home/ec2-user/single-winner-generative-social-choice

TOPICS="abortion healthcare electoral policing trust environment"
LOG_FILE="logs/clustered_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "Starting clustered pipeline at $(date)" | tee -a "$LOG_FILE"

# Phase 1: Conservative (rep1-rep9)
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 1: CONSERVATIVE_TRADITIONAL" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for topic in $TOPICS; do
  for rep in {1..9}; do
    echo "=== Conservative: $topic rep$rep ===" | tee -a "$LOG_FILE"
    uv run python -m src.sample_alt_voters.run_experiment \
      --voter-dist conservative_traditional --topic $topic \
      --alt-dist persona_no_context --rep $rep 2>&1 | tee -a "$LOG_FILE"
  done
done

echo "Conservative data complete at $(date)" | tee -a "$LOG_FILE"

# Phase 2: GPT insertion for conservative
echo "" | tee -a "$LOG_FILE"
echo "Running GPT insertion for conservative reps..." | tee -a "$LOG_FILE"
uv run python scripts/run_gpt_star_batched.py --skip-completed 2>&1 | tee -a "$LOG_FILE"

echo "Conservative GPT insertion complete at $(date)" | tee -a "$LOG_FILE"

# Phase 3: Progressive (rep1-rep9)
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 2: PROGRESSIVE_LIBERAL" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for topic in $TOPICS; do
  for rep in {1..9}; do
    echo "=== Progressive: $topic rep$rep ===" | tee -a "$LOG_FILE"
    uv run python -m src.sample_alt_voters.run_experiment \
      --voter-dist progressive_liberal --topic $topic \
      --alt-dist persona_no_context --rep $rep 2>&1 | tee -a "$LOG_FILE"
  done
done

echo "Progressive data complete at $(date)" | tee -a "$LOG_FILE"

# Phase 4: GPT insertion for progressive
echo "" | tee -a "$LOG_FILE"
echo "Running GPT insertion for progressive reps..." | tee -a "$LOG_FILE"
uv run python scripts/run_gpt_star_batched.py --skip-completed 2>&1 | tee -a "$LOG_FILE"

echo "Progressive GPT insertion complete at $(date)" | tee -a "$LOG_FILE"

# Phase 5: Regenerate plots
echo "" | tee -a "$LOG_FILE"
echo "Regenerating visualizations..." | tee -a "$LOG_FILE"
uv run python scripts/generate_slide_plots.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
