#!/bin/bash
# Complete test run: archive old data, run experiments, generate reports

set -e

echo "🔄 COMPLETE TEST RUN PIPELINE"
echo "================================"
echo ""

# 1. Archive existing data if it exists
if [ "$(ls -A data/large_scale/results 2>/dev/null)" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_DIR="test_runs/complete_run_${TIMESTAMP}"
    
    echo "📦 Step 1: Archiving existing data..."
    mkdir -p "$ARCHIVE_DIR"
    cp -r data/large_scale "$ARCHIVE_DIR/"
    
    if [ -d outputs/test_run_* ] || [ -d outputs/prod_run_* ]; then
        mkdir -p "$ARCHIVE_DIR/previous_outputs"
        cp -r outputs/*_run_* "$ARCHIVE_DIR/previous_outputs/" 2>/dev/null || true
    fi
    
    echo "   ✓ Archived to: $ARCHIVE_DIR"
    echo ""
fi

# 2. Clear existing test results
echo "🗑️  Step 2: Clearing previous test results..."
rm -rf data/large_scale/test/statements/* \
       data/large_scale/test/preferences/* \
       data/large_scale/test/evaluations/* \
       data/large_scale/test/results/* 2>/dev/null || true
echo "   ✓ Cleared"
echo ""

# 3. Run experiments
echo "🚀 Step 3: Running experiments (3 topics)..."
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
for TOPIC_IDX in 0 1 2; do
    echo "   Running topic $TOPIC_IDX..."
    python -m src.large_scale.main \
        --test-mode \
        --load-personas \
        --topic-index $TOPIC_IDX \
        2>&1 | tee "logs/test_run_${RUN_TIMESTAMP}_topic${TOPIC_IDX}.log" > /dev/null
    echo "   ✓ Topic $TOPIC_IDX completed"
done
echo ""

# 4. Generate reports
echo "📊 Step 4: Generating reports..."
bash scripts/generate_timestamped_reports.sh test
echo ""

echo "✅ COMPLETE TEST RUN FINISHED!"
echo ""
echo "📁 View results:"
LATEST_OUTPUT=$(ls -td outputs/test_run_* 2>/dev/null | head -1)
echo "   Latest reports: $LATEST_OUTPUT"
echo "   Results JSON:   data/large_scale/test/results/"
echo "   Logs:           logs/test_run_${RUN_TIMESTAMP}_*.log"

