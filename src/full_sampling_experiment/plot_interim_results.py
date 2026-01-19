"""
Plot interim results from the full sampling experiment.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "full_sampling_experiment"
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Method display names and colors
METHOD_DISPLAY = {
    "schulze": "Schulze",
    "borda": "Borda",
    "irv": "IRV",
    "plurality": "Plurality",
    "veto_by_consumption": "Veto by Consumption",
    "chatgpt": "ChatGPT",
    "chatgpt_rankings": "ChatGPT+Rankings",
    "chatgpt_personas": "ChatGPT+Personas",
    "chatgpt_star": "ChatGPT*",
    "chatgpt_star_rankings": "ChatGPT*+Rankings",
    "chatgpt_star_personas": "ChatGPT*+Personas",
    "chatgpt_double_star": "ChatGPT**",
    "chatgpt_double_star_rankings": "ChatGPT**+Rankings",
    "chatgpt_double_star_personas": "ChatGPT**+Personas",
    "chatgpt_triple_star": "ChatGPT***",
}

# Color scheme
COLORS = {
    # Traditional methods - blues/greens
    "schulze": "#1f77b4",
    "borda": "#2ca02c",
    "irv": "#17becf",
    "plurality": "#8c564b",
    "veto_by_consumption": "#9467bd",
    # ChatGPT variants - oranges/reds
    "chatgpt": "#ff7f0e",
    "chatgpt_rankings": "#ffbb78",
    "chatgpt_personas": "#ffd700",
    "chatgpt_star": "#d62728",
    "chatgpt_star_rankings": "#ff9896",
    "chatgpt_star_personas": "#e377c2",
    "chatgpt_double_star": "#7f7f7f",
    "chatgpt_double_star_rankings": "#c7c7c7",
    "chatgpt_double_star_personas": "#bcbd22",
    "chatgpt_triple_star": "#000000",
}

# Short topic names for display
TOPIC_SHORT_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "Trust",
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


def load_all_results():
    """Load results from all completed topics."""
    all_results = defaultdict(list)
    topic_results = {}
    
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        topic_name = topic_dir.name
        topic_results[topic_name] = defaultdict(list)
        
        for rep_dir in sorted(topic_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep"):
                continue
            
            for sample_dir in sorted(rep_dir.iterdir()):
                if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
                    continue
                
                results_file = sample_dir / "results.json"
                if not results_file.exists():
                    continue
                
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    
                    for method, data in results.items():
                        if "epsilon" in data:
                            eps = data["epsilon"]
                            all_results[method].append(eps)
                            topic_results[topic_name][method].append(eps)
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
    
    return all_results, topic_results


def plot_cdf_all_methods(all_results, output_path):
    """Plot CDF for all methods."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    methods_order = [
        "schulze", "borda", "irv", "plurality", "veto_by_consumption",
        "chatgpt", "chatgpt_rankings", "chatgpt_personas",
        "chatgpt_star", "chatgpt_star_rankings", "chatgpt_star_personas",
        "chatgpt_double_star", "chatgpt_double_star_rankings", "chatgpt_double_star_personas",
        "chatgpt_triple_star"
    ]
    
    for method in methods_order:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        
        epsilons = sorted(all_results[method])
        n = len(epsilons)
        cdf = np.arange(1, n + 1) / n
        
        ax.step(epsilons, cdf, where='post', label=METHOD_DISPLAY.get(method, method),
                color=COLORS.get(method, '#333333'), linewidth=2)
    
    ax.set_xlabel("Critical Epsilon (ε*)", fontsize=14)
    ax.set_ylabel("Cumulative Probability", fontsize=14)
    ax.set_title("CDF of Critical Epsilon by Voting Method (All Topics)", fontsize=16)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cdf_grouped(all_results, output_path):
    """Plot CDF with methods grouped: Traditional vs LLM-based."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    traditional = ["schulze", "borda", "irv", "plurality", "veto_by_consumption"]
    llm_based = ["chatgpt", "chatgpt_star", "chatgpt_double_star", "chatgpt_triple_star"]
    
    # Traditional methods
    ax = axes[0]
    for method in traditional:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = sorted(all_results[method])
        n = len(epsilons)
        cdf = np.arange(1, n + 1) / n
        ax.step(epsilons, cdf, where='post', label=METHOD_DISPLAY.get(method, method),
                color=COLORS.get(method), linewidth=2.5)
    
    ax.set_xlabel("Critical Epsilon (ε*)", fontsize=14)
    ax.set_ylabel("Cumulative Probability", fontsize=14)
    ax.set_title("Traditional Voting Methods", fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1)
    
    # LLM-based methods
    ax = axes[1]
    for method in llm_based:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = sorted(all_results[method])
        n = len(epsilons)
        cdf = np.arange(1, n + 1) / n
        ax.step(epsilons, cdf, where='post', label=METHOD_DISPLAY.get(method, method),
                color=COLORS.get(method), linewidth=2.5)
    
    ax.set_xlabel("Critical Epsilon (ε*)", fontsize=14)
    ax.set_ylabel("Cumulative Probability", fontsize=14)
    ax.set_title("LLM-Based Methods", fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_mean_epsilon(all_results, output_path):
    """Plot bar chart of mean epsilon by method."""
    methods_order = [
        "schulze", "borda", "irv", "plurality", "veto_by_consumption",
        "chatgpt", "chatgpt_star", "chatgpt_double_star", "chatgpt_triple_star"
    ]
    
    means = []
    stds = []
    labels = []
    colors = []
    
    for method in methods_order:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = all_results[method]
        means.append(np.mean(epsilons))
        stds.append(np.std(epsilons))
        labels.append(METHOD_DISPLAY.get(method, method))
        colors.append(COLORS.get(method, '#333333'))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel("Mean Critical Epsilon (ε*)", fontsize=14)
    ax.set_title("Mean Critical Epsilon by Voting Method", fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_topic(topic_results, output_path):
    """Plot mean epsilon by topic for key methods."""
    key_methods = ["schulze", "veto_by_consumption", "chatgpt_double_star", "chatgpt_triple_star"]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    topics = list(topic_results.keys())
    x = np.arange(len(topics))
    width = 0.2
    
    for i, method in enumerate(key_methods):
        means = []
        for topic in topics:
            if method in topic_results[topic] and len(topic_results[topic][method]) > 0:
                means.append(np.mean(topic_results[topic][method]))
            else:
                means.append(0)
        
        ax.bar(x + i * width, means, width, label=METHOD_DISPLAY.get(method, method),
               color=COLORS.get(method, '#333333'), alpha=0.8)
    
    ax.set_xticks(x + width * 1.5)
    short_names = [TOPIC_SHORT_NAMES.get(t, t[:20]) for t in topics]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("Mean Critical Epsilon (ε*)", fontsize=14)
    ax.set_title("Critical Epsilon by Topic and Method", fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_stats(all_results, output_path):
    """Create a summary statistics table as a figure."""
    methods_order = [
        "schulze", "borda", "irv", "plurality", "veto_by_consumption",
        "chatgpt", "chatgpt_star", "chatgpt_double_star", "chatgpt_triple_star"
    ]
    
    data = []
    for method in methods_order:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = all_results[method]
        data.append([
            METHOD_DISPLAY.get(method, method),
            len(epsilons),
            f"{np.mean(epsilons):.4f}",
            f"{np.std(epsilons):.4f}",
            f"{np.median(epsilons):.4f}",
            f"{np.min(epsilons):.4f}",
            f"{np.max(epsilons):.4f}",
            f"{100 * np.mean(np.array(epsilons) == 0):.1f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ["Method", "N", "Mean", "Std", "Median", "Min", "Max", "% ε=0"]
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Summary Statistics: Critical Epsilon by Method", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading results...")
    all_results, topic_results = load_all_results()
    
    print(f"\nLoaded data for {len(topic_results)} topics")
    for method, epsilons in all_results.items():
        print(f"  {method}: {len(epsilons)} samples")
    
    print("\nGenerating plots...")
    
    # CDF plots
    plot_cdf_all_methods(all_results, FIGURES_DIR / "cdf_all_methods.png")
    plot_cdf_grouped(all_results, FIGURES_DIR / "cdf_grouped.png")
    
    # Bar chart
    plot_bar_mean_epsilon(all_results, FIGURES_DIR / "bar_mean_epsilon.png")
    
    # By topic
    plot_by_topic(topic_results, FIGURES_DIR / "by_topic.png")
    
    # Summary stats
    plot_summary_stats(all_results, FIGURES_DIR / "summary_stats.png")
    
    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
