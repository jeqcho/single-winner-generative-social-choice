#!/bin/bash
# Monitor production run progress

LOG_FILE=$(ls -t logs/prod_gpt5nano_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No production log file found"
    exit 1
fi

echo "📊 Monitoring: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show current status
echo "🔍 Current Status:"
echo ""

# Count completed topics
COMPLETED=$(ls -1 data/large_scale/prod/results/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  Topics completed: $COMPLETED/13"
echo ""

# Show recent log entries
echo "📝 Recent log entries:"
echo ""
tail -30 "$LOG_FILE"
echo ""

# Check for errors
ERRORS=$(grep -i "error\|exception\|failed" "$LOG_FILE" | tail -5)
if [ ! -z "$ERRORS" ]; then
    echo "⚠️  Recent errors:"
    echo "$ERRORS"
else
    echo "✅ No recent errors"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "To continue monitoring:"
echo "  tail -f $LOG_FILE"
echo "  watch -n 60 './scripts/monitor_production.sh'"

