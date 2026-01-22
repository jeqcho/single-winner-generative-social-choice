#!/bin/bash
# Generate Alt2 and Alt3 statements for all reps
# This will make ~2,400 API calls total:
# - Alt2: 100 personas × 10 reps × 2 topics = 2,000 API calls
# - Alt3: 20 batches × 10 reps × 2 topics = 400 API calls

set -e

cd /home/ec2-user/single-winner-generative-social-choice

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/per_rep_generation_${TIMESTAMP}.log"

echo "=== Starting per-rep generation at $(date) ===" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "Estimated: ~2,400 API calls (Alt2: 2,000 + Alt3: 400)" | tee -a "$LOG_FILE"

uv run python -m src.sample_alt_voters.generate_per_rep_statements --all 2>&1 | tee -a "$LOG_FILE"

echo "=== Per-rep generation complete at $(date) ===" | tee -a "$LOG_FILE"
