#!/bin/bash
# Quick test script to run 2 topics

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "🚀 Running 2-topic test with Responses API (gpt-5.1)"
echo "📝 Logs will be saved to: $LOG_DIR/run_${TIMESTAMP}.log"
echo ""

# Run topics 0 and 1
for TOPIC_IDX in 0 1; do
    echo "Running topic index $TOPIC_IDX..."
    python -m src.large_scale.main --test-mode --load-personas --topic-index $TOPIC_IDX 2>&1 | tee -a "$LOG_DIR/run_${TIMESTAMP}.log"
    echo "✓ Topic $TOPIC_IDX completed"
    echo ""
done

echo "✅ Test completed! Check results in data/large_scale/results/"
echo "📊 Generate reports with:"
echo "  python -m src.large_scale.generate_pvc_table"
echo "  python -m src.large_scale.generate_pvc_size_table"
echo "  python -m src.large_scale.generate_technique_histograms"


