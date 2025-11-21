#!/bin/bash
# Production run with gpt-5-nano (950/50/50 personas)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PRODUCTION RUN WITH GPT-5-NANO (950/50/50 personas)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Started at: $(date)"
echo ""

# Create timestamped log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/prod_gpt5nano.log"

# Step 1: Load/generate personas
echo "📦 Step 1: Loading personas (900/50/50)..."
python -m src.large_scale.persona_loader \
  --n-generative 900 \
  --n-discriminative 50 \
  --n-evaluative 50 \
  2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Persona loading failed!"
    exit 1
fi

echo ""
echo "✅ Personas loaded successfully"
echo ""

# Step 2: Run experiments (production mode = no --test-mode flag)
echo "🔬 Step 2: Running experiments on 13 topics..."
echo "   This will take several hours..."
echo "   Monitor progress: tail -f $LOG_FILE"
echo ""

python -m src.large_scale.main \
  --load-personas \
  2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Experiment failed!"
    exit 1
fi

echo ""
echo "✅ All experiments completed"
echo ""

# Step 3: Generate reports
echo "📊 Step 3: Generating reports..."
echo ""

# Generate PVC tables
echo "  → Generating PVC winner table..."
python -m src.large_scale.generate_pvc_table \
  --results-dir data/large_scale/prod/results \
  --latex-output "outputs/prod_run_${TIMESTAMP}/tables/pvc_winner_table.tex" \
  --csv-output "outputs/prod_run_${TIMESTAMP}/tables/pvc_winner_table.csv" \
  2>&1 | tee -a "$LOG_FILE"

echo "  → Generating PVC size table..."
python -m src.large_scale.generate_pvc_size_table \
  --results-dir data/large_scale/prod/results \
  --latex-output "outputs/prod_run_${TIMESTAMP}/tables/pvc_size_table.tex" \
  --csv-output "outputs/prod_run_${TIMESTAMP}/tables/pvc_size_table.csv" \
  2>&1 | tee -a "$LOG_FILE"

# Generate histograms
echo "  → Generating method histograms..."
python -m src.large_scale.generate_technique_histograms \
  --results-dir data/large_scale/prod/results \
  --output-dir "outputs/prod_run_${TIMESTAMP}/figures" \
  2>&1 | tee -a "$LOG_FILE"

# Generate average Likert chart
echo "  → Generating average Likert comparison chart..."
python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("data/large_scale/prod/results")
results = [json.load(open(f)) for f in sorted(results_dir.glob("*.json"))]

methods = [
    ("plurality", "Plurality"),
    ("borda", "Borda"),
    ("irv", "IRV"),
    ("rankedpairs", "RankedPairs"),
    ("successive_veto", "Successive Veto"),
    ("chatgpt", "ChatGPT"),
    ("chatgpt_rankings", "ChatGPT+Rankings"),
    ("chatgpt_profiles", "ChatGPT+Profiles"),
    ("chatgpt_rankings_profiles", "ChatGPT+R+P"),
]

method_avg_ratings, method_labels, method_stds, method_counts = [], [], [], []

for method_key, method_label in methods:
    all_ratings = []
    for result in results:
        method_result = result["method_results"].get(method_key)
        if not method_result or "error" in method_result:
            continue
        winner_idx = method_result.get("winner")
        if winner_idx is None:
            continue
        winner_idx = int(winner_idx)
        for eval_item in result["evaluations"]:
            ratings = eval_item["ratings"]
            if winner_idx < len(ratings):
                all_ratings.append(ratings[winner_idx])
    
    if all_ratings:
        method_avg_ratings.append(np.mean(all_ratings))
        method_stds.append(np.std(all_ratings))
        method_labels.append(method_label)
        method_counts.append(len(all_ratings))

fig, ax = plt.subplots(figsize=(12, 8))
x_pos = np.arange(len(method_labels))
ax.bar(x_pos, method_avg_ratings, yerr=method_stds, capsize=5,
       alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Voting Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Likert Score', fontsize=14, fontweight='bold')
ax.set_title('Average Evaluative Ratings by Voting Method (Across All Topics)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=11)
ax.set_ylim(0, 5.5)
ax.grid(axis='y', alpha=0.3)

for i, (avg, std) in enumerate(zip(method_avg_ratings, method_stds)):
    ax.text(i, avg + std + 0.1, f'{avg:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

summary_text = f'N = {len(results)} topics, {method_counts[0]} ratings per method'
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

Path("outputs/prod_run_${TIMESTAMP}/figures").mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig("outputs/prod_run_${TIMESTAMP}/figures/average_likert_by_method.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Average Likert chart saved")
EOF

echo ""
echo "✅ All reports generated"
echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 PRODUCTION RUN COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Completed at: $(date)"
echo ""
echo "📁 Results location:"
echo "   Data:    data/large_scale/prod/results/"
echo "   Tables:  outputs/prod_run_${TIMESTAMP}/tables/"
echo "   Figures: outputs/prod_run_${TIMESTAMP}/figures/"
echo "   Log:     $LOG_FILE"
echo ""
echo "📊 Generated files:"
ls -lh "outputs/prod_run_${TIMESTAMP}/tables/" 2>/dev/null
echo ""
ls -lh "outputs/prod_run_${TIMESTAMP}/figures/" 2>/dev/null
echo ""

