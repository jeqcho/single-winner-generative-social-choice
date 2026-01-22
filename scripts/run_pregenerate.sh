#!/bin/bash
# Pre-generate Alt1 and Alt4 statements for sample-alt-voters experiment
# This will make ~1956 API calls total

set -e

cd /home/ec2-user/single-winner-generative-social-choice

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pregenerate_${TIMESTAMP}.log"

echo "=== Starting pre-generation at $(date) ===" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"

# Alt1 for abortion (815 API calls)
echo "=== Generating Alt1 for abortion ===" | tee -a "$LOG_FILE"
uv run python -m src.sample_alt_voters.generate_statements --alt1 --topic abortion 2>&1 | tee -a "$LOG_FILE"

# Alt1 for electoral (815 API calls)
echo "=== Generating Alt1 for electoral ===" | tee -a "$LOG_FILE"
uv run python -m src.sample_alt_voters.generate_statements --alt1 --topic electoral 2>&1 | tee -a "$LOG_FILE"

# Alt4 for abortion (163 API calls)
echo "=== Generating Alt4 for abortion ===" | tee -a "$LOG_FILE"
uv run python -m src.sample_alt_voters.generate_statements --alt4 --topic abortion --n 815 2>&1 | tee -a "$LOG_FILE"

# Alt4 for electoral (163 API calls)
echo "=== Generating Alt4 for electoral ===" | tee -a "$LOG_FILE"
uv run python -m src.sample_alt_voters.generate_statements --alt4 --topic electoral --n 815 2>&1 | tee -a "$LOG_FILE"

echo "=== Pre-generation complete at $(date) ===" | tee -a "$LOG_FILE"
