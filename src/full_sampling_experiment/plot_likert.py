"""
Plot Likert score histograms for all topics.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "full_sampling_experiment"
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Short topic names for display
TOPIC_SHORT_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "Public Trust",
    "what-are-the-best-policies-to-prevent-littering-in": "Littering",
    "what-are-your-thoughts-on-the-way-university-campu": "Campus Speech",
    "what-balance-should-be-struck-between-environmenta": "Environment",
    "what-balance-should-exist-between-gun-safety-laws-": "Gun Safety",
    "what-limits-if-any-should-exist-on-free-speech-reg": "Free Speech",
    "what-principles-should-guide-immigration-policy-an": "Immigration",
    "what-reforms-if-any-should-replace-or-modify-the-e": "Electoral College",
    "what-role-if-any-should-race-and-identity-play-in-": "Race/Identity",
    "what-role-should-the-government-play-in-regulating": "Tech Regulation",
    "what-should-be-done-about-abortion-policy-in-the-u": "Abortion",
    "what-should-be-done-to-address-racial-and-ethnic-i": "Racial Inequality",
    "what-should-be-the-role-of-government-in-reducing-": "Inequality",
}


def load_likert_scores():
    """Load all Likert scores from completed topics."""
    topic_scores = {}
    
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        topic_name = topic_dir.name
        likert_file = topic_dir / "rep0" / "likert_scores.json"
        
        if likert_file.exists():
            with open(likert_file) as f:
                data = json.load(f)
            # Flatten all scores (100 voters x 100 statements = 10000 scores per topic)
            all_scores = np.array(data["scores"]).flatten()
            topic_scores[topic_name] = all_scores
            print(f"Loaded {len(all_scores)} scores for {topic_name}")
    
    return topic_scores


def plot_likert_histograms(topic_scores, output_path):
    """Plot histogram for each topic in a single figure."""
    n_topics = len(topic_scores)
    
    # Create figure with one row per topic
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 2.5 * n_topics))
    
    if n_topics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
    
    for ax, (topic_name, scores), color in zip(axes, topic_scores.items(), colors):
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        
        # Create histogram with bins 1-10
        bins = np.arange(0.5, 11.5, 1)  # 0.5, 1.5, 2.5, ..., 10.5
        counts, _ = np.histogram(scores, bins=bins)
        normalized = counts / len(scores)
        
        ax.bar(range(1, 11), normalized, color=color, edgecolor='black', alpha=0.8)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, max(normalized) * 1.15)
        ax.set_xticks(range(1, 11))
        ax.set_title(short_name, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel("Likert Score (1=Strongly Disagree, 10=Strongly Agree)", fontsize=12)
    
    plt.suptitle("Distribution of Likert Agreement Scores by Topic\n(100 voters × 100 statements per topic)", 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_likert_combined(topic_scores, output_path):
    """Plot all topics overlaid on one histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(topic_scores)))
    bins = np.arange(0.5, 11.5, 1)
    
    for (topic_name, scores), color in zip(topic_scores.items(), colors):
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:20])
        counts, _ = np.histogram(scores, bins=bins)
        normalized = counts / len(scores)
        ax.plot(range(1, 11), normalized, marker='o', linewidth=2, markersize=6,
                color=color, label=short_name, alpha=0.8)
    
    ax.set_xlabel("Likert Score (1=Strongly Disagree, 10=Strongly Agree)", fontsize=12)
    ax.set_ylabel("Normalized Frequency", fontsize=12)
    ax.set_title("Likert Agreement Score Distribution by Topic", fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_likert_summary(topic_scores, output_path):
    """Create summary statistics for Likert scores."""
    data = []
    for topic_name, scores in topic_scores.items():
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        data.append([
            short_name,
            len(scores),
            f"{np.mean(scores):.2f}",
            f"{np.std(scores):.2f}",
            f"{np.median(scores):.1f}",
            f"{100 * np.mean(scores >= 7):.1f}%",  # % agree (7-10)
            f"{100 * np.mean(scores <= 4):.1f}%",  # % disagree (1-4)
        ])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    columns = ["Topic", "N", "Mean", "Std", "Median", "% Agree (≥7)", "% Disagree (≤4)"]
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Likert Score Summary Statistics by Topic", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading Likert scores...")
    topic_scores = load_likert_scores()
    
    if not topic_scores:
        print("No Likert data found!")
        return
    
    print(f"\nLoaded data for {len(topic_scores)} topics")
    
    print("\nGenerating plots...")
    plot_likert_histograms(topic_scores, FIGURES_DIR / "likert_histograms.png")
    plot_likert_combined(topic_scores, FIGURES_DIR / "likert_combined.png")
    plot_likert_summary(topic_scores, FIGURES_DIR / "likert_summary.png")
    
    print(f"\nAll Likert plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
