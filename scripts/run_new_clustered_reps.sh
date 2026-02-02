#!/bin/bash
# Run 9 new reps (rep1-rep9) for both ideology clusters
# 6 topics × 2 voter_dists × 9 reps = 108 new conditions

cd /home/ec2-user/single-winner-generative-social-choice

TOPICS="abortion healthcare electoral policing trust environment"
LOG_FILE="logs/new_clustered_reps_$(date +%Y%m%d_%H%M%S).log"

echo "Starting new clustered reps at $(date)" | tee -a "$LOG_FILE"

for topic in $TOPICS; do
  for rep in {1..9}; do
    echo "=== Running $topic rep$rep ===" | tee -a "$LOG_FILE"
    
    echo "  Progressive liberal..." | tee -a "$LOG_FILE"
    uv run python -m src.sample_alt_voters.run_experiment \
      --voter-dist progressive_liberal --topic $topic \
      --alt-dist persona_no_context --rep $rep 2>&1 | tee -a "$LOG_FILE"
    
    echo "  Conservative traditional..." | tee -a "$LOG_FILE"
    uv run python -m src.sample_alt_voters.run_experiment \
      --voter-dist conservative_traditional --topic $topic \
      --alt-dist persona_no_context --rep $rep 2>&1 | tee -a "$LOG_FILE"
  done
done

echo "Completed at $(date)" | tee -a "$LOG_FILE"
