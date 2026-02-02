#!/usr/bin/env python3
"""Generate paper-quality plots comparing generative ChatGPT methods to VBC."""
import json
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from src.sample_alt_voters.results_aggregator import collect_all_results

PAPER_OUTPUT_DIR = project_root / "outputs" / "paper" / "plots"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

METHODS_TO_COMPARE = ["veto_by_consumption", "chatgpt_double_star", "chatgpt_double_star_rankings", "chatgpt_double_star_personas", "chatgpt_triple_star", "random_insertion"]
METHOD_LABELS = {"veto_by_consumption": "VBC", "chatgpt_double_star": "GPT**", "chatgpt_double_star_rankings": "GPT**+Rank", "chatgpt_double_star_personas": "GPT**+Pers", "chatgpt_triple_star": "GPT***", "random_insertion": "Random Insertion", "random": "Random"}
METHOD_ORDER = list(METHODS_TO_COMPARE) + ["random"]
COLORS = {"veto_by_consumption": "#0072B2", "chatgpt_double_star": "#D55E00", "chatgpt_double_star_rankings": "#CC79A7", "chatgpt_double_star_personas": "#009E73", "chatgpt_triple_star": "#E69F00", "random_insertion": "#999999", "random": "#000000"}
TOPIC_DISPLAY_NAMES = {"abortion": "Abortion", "electoral": "Electoral College", "healthcare": "Healthcare", "policing": "Policing", "environment": "Environment", "trust": "Trust in Institutions"}
ALL_TOPICS = ["abortion", "electoral", "healthcare", "policing", "environment", "trust"]

def load_and_filter_data():
    """Load experiment results and random baseline data."""
    df = collect_all_results()
    filtered_df = df[(df["voter_dist"] == "uniform") & (df["alt_dist"] == "persona_no_context")].copy()
    filtered_df = filtered_df[filtered_df["method"].isin(METHODS_TO_COMPARE)]
    
    # Load random baseline (precomputed epsilons for all alternatives)
    random_df = load_random_baseline()
    if not random_df.empty:
        # Combine with main data
        filtered_df = pd.concat([filtered_df, random_df], ignore_index=True)
    
    return filtered_df

def get_method_label(m): return METHOD_LABELS.get(m, m)
def get_method_color(m): return COLORS.get(m, "#666666")
def compute_ci(data, conf=0.95):
    if len(data) == 0: return (np.nan, np.nan)
    n, mean, se = len(data), np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return (mean - h, mean + h)

def load_random_baseline(topics=None, n_reps=10):
    """Load precomputed epsilons to create Random baseline data.
    
    This represents truly random selection from the alternative pool -
    the epsilon values for all 100 alternatives if they were selected.
    """
    if topics is None:
        topics = ALL_TOPICS
    random_data = []
    for topic in topics:
        for rep in range(n_reps):
            path = project_root / f"outputs/sample_alt_voters/data/{topic}/uniform/persona_no_context/rep{rep}/precomputed_epsilons.json"
            if path.exists():
                with open(path) as f:
                    eps_dict = json.load(f)
                    for alt_id, epsilon in eps_dict.items():
                        random_data.append({
                            "topic": topic,
                            "rep_id": rep,
                            "mini_rep_id": 0,  # Placeholder for compatibility
                            "method": "random",
                            "epsilon": epsilon,
                            "alt_id": alt_id
                        })
    return pd.DataFrame(random_data)

def plot_cdf_vbc_vs_gpt(df, topic, output_path, x_max=0.5, y_min=0.5, zoomed=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
    if topic_df.empty:
        plt.close()
        return
    for method in METHOD_ORDER:
        method_data = topic_df[topic_df["method"] == method]["epsilon"].values
        if len(method_data) == 0:
            continue
        sorted_data = np.sort(method_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        lw = 3 if method == "veto_by_consumption" else 2
        ls = "--" if method == "random" else "-"  # Dashed line for Random baseline
        ax.step(sorted_data, cdf, where="post", label=get_method_label(method), color=get_method_color(method), linewidth=lw, linestyle=ls)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, 1.02)
    ax.set_xlabel("Critical Epsilon")
    ax.set_ylabel("Cumulative Probability")
    zoom_str = " (Zoomed)" if zoomed else ""
    ax.set_title(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: VBC vs GPT{zoom_str}")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_boxplot(df, topic, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
    if topic_df.empty:
        plt.close()
        return
    plot_data, labels, colors, methods_used = [], [], [], []
    for method in METHOD_ORDER:
        method_data = topic_df[topic_df["method"] == method]["epsilon"].values
        if len(method_data) > 0:
            plot_data.append(method_data)
            labels.append(get_method_label(method))
            colors.append(get_method_color(method))
            methods_used.append(method)
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
    for i, (patch, color, method) in enumerate(zip(bp["boxes"], colors, methods_used)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        if method == "random":
            patch.set_hatch("//")  # Hatched pattern for Random baseline
    ax.set_ylabel("Critical Epsilon")
    ax.set_xlabel("Method")
    ax.set_title(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: Epsilon Distribution by Method")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_scatter_vbc_vs_gpt(df, topic, output_path):
    gpt_methods = [m for m in METHOD_ORDER if m not in ["veto_by_consumption", "random_insertion", "random"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
    if topic_df.empty:
        plt.close()
        return
    vbc_df = topic_df[topic_df["method"] == "veto_by_consumption"].set_index(["rep_id", "mini_rep_id"])
    for idx, gpt_method in enumerate(gpt_methods):
        ax = axes[idx]
        gpt_df = topic_df[topic_df["method"] == gpt_method].set_index(["rep_id", "mini_rep_id"])
        common_idx = vbc_df.index.intersection(gpt_df.index)
        if len(common_idx) == 0:
            ax.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=ax.transAxes)
            continue
        vbc_eps = vbc_df.loc[common_idx, "epsilon"].values
        gpt_eps = gpt_df.loc[common_idx, "epsilon"].values
        jitter = 0.005
        ax.scatter(vbc_eps + np.random.uniform(-jitter, jitter, len(vbc_eps)), gpt_eps + np.random.uniform(-jitter, jitter, len(gpt_eps)), alpha=0.5, s=30, color=get_method_color(gpt_method))
        max_val = max(vbc_eps.max(), gpt_eps.max(), 0.3)
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5)
        gpt_wins = np.sum(gpt_eps < vbc_eps)
        vbc_wins = np.sum(vbc_eps < gpt_eps)
        ties = np.sum(vbc_eps == gpt_eps)
        ax.set_xlabel("VBC Epsilon")
        ax.set_ylabel(f"{get_method_label(gpt_method)} Epsilon")
        ax.set_title(f"{get_method_label(gpt_method)} vs VBC\n(GPT wins:{gpt_wins}, VBC wins:{vbc_wins}, Ties:{ties})")
        ax.set_xlim(-0.02, max_val + 0.02)
        ax.set_ylim(-0.02, max_val + 0.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: Paired Epsilon Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_bar_epsilon(df, topic, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
    if topic_df.empty:
        plt.close()
        return
    means, errors_low, errors_high, labels, colors, methods_used = [], [], [], [], [], []
    for method in METHOD_ORDER:
        method_data = topic_df[topic_df["method"] == method]["epsilon"].values
        if len(method_data) > 0:
            mean = np.mean(method_data)
            ci_low, ci_high = compute_ci(method_data)
            means.append(mean)
            errors_low.append(mean - ci_low)
            errors_high.append(ci_high - mean)
            labels.append(get_method_label(method))
            colors.append(get_method_color(method))
            methods_used.append(method)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=[errors_low, errors_high], capsize=5, color=colors, alpha=0.7, edgecolor="black")
    # Add hatch pattern for Random baseline
    for i, (bar, method) in enumerate(zip(bars, methods_used)):
        if method == "random":
            bar.set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Critical Epsilon")
    ax.set_xlabel("Method")
    ax.set_title(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: Mean Epsilon with 95% CI")
    all_mean = topic_df["epsilon"].mean()
    ax.axhline(y=all_mean, color="black", linestyle="--", linewidth=1.5, label=f"Overall mean={all_mean:.3f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def generate_summary_table(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for method in METHOD_ORDER:
        method_data = df[df["method"] == method]["epsilon"].dropna().values
        if len(method_data) == 0:
            continue
        mean = np.mean(method_data)
        std = np.std(method_data)
        median = np.median(method_data)
        ci_low, ci_high = compute_ci(method_data)
        pct_zero = 100 * np.sum(method_data == 0) / len(method_data)
        summary_rows.append({"Method": get_method_label(method), "N": len(method_data), "Mean": mean, "Std": std, "Median": median, "95% CI Low": ci_low, "95% CI High": ci_high, "% Zero": pct_zero})
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "summary_by_method.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")
    latex_path = output_dir / "summary_by_method.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Summary Statistics by Method}\n\\label{tab:summary_method}\n\\begin{tabular}{lcccccc}\n\\toprule\nMethod & N & Mean & Std & Median & 95\\% CI & \\% Zero \\\\\n\\midrule\n")
        for _, row in summary_df.iterrows():
            ci_str = f"[{row['95% CI Low']:.3f}, {row['95% CI High']:.3f}]"
            f.write(f"{row['Method']} & {row['N']} & {row['Mean']:.4f} & {row['Std']:.4f} & {row['Median']:.4f} & {ci_str} & {row['% Zero']:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {latex_path}")
    topic_rows = []
    for topic in ALL_TOPICS:
        topic_df = df[df["topic"] == topic]
        for method in METHOD_ORDER:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].dropna().values
            if len(method_data) == 0:
                continue
            topic_rows.append({"Topic": TOPIC_DISPLAY_NAMES.get(topic, topic), "Method": get_method_label(method), "N": len(method_data), "Mean": np.mean(method_data), "Std": np.std(method_data), "Median": np.median(method_data), "% Zero": 100 * np.sum(method_data == 0) / len(method_data)})
    topic_summary_df = pd.DataFrame(topic_rows)
    topic_csv_path = output_dir / "summary_by_topic_method.csv"
    topic_summary_df.to_csv(topic_csv_path, index=False, float_format="%.4f")
    print(f"Saved: {topic_csv_path}")

def plot_heatmap_method_topic(df, output_path):
    pivot_data = []
    for topic in ALL_TOPICS:
        topic_df = df[df["topic"] == topic]
        row = {"Topic": TOPIC_DISPLAY_NAMES.get(topic, topic)}
        for method in METHOD_ORDER:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].dropna().values
            row[get_method_label(method)] = np.mean(method_data) if len(method_data) > 0 else np.nan
        pivot_data.append(row)
    pivot_df = pd.DataFrame(pivot_data).set_index("Topic")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Mean Epsilon"}, linewidths=0.5, linecolor="white")
    ax.set_title("Mean Epsilon by Topic and Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_win_tie_loss(df, output_path):
    gpt_methods = [m for m in METHOD_ORDER if m not in ["veto_by_consumption", "random_insertion", "random"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    win_counts, tie_counts, loss_counts, method_labels = [], [], [], []
    for gpt_method in gpt_methods:
        wins, ties, losses = 0, 0, 0
        for topic in ALL_TOPICS:
            topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
            vbc_df = topic_df[topic_df["method"] == "veto_by_consumption"].set_index(["rep_id", "mini_rep_id"])
            gpt_df = topic_df[topic_df["method"] == gpt_method].set_index(["rep_id", "mini_rep_id"])
            common_idx = vbc_df.index.intersection(gpt_df.index)
            if len(common_idx) == 0:
                continue
            vbc_eps = vbc_df.loc[common_idx, "epsilon"].values
            gpt_eps = gpt_df.loc[common_idx, "epsilon"].values
            wins += np.sum(gpt_eps < vbc_eps)
            losses += np.sum(gpt_eps > vbc_eps)
            ties += np.sum(gpt_eps == vbc_eps)
        total = wins + ties + losses
        if total > 0:
            win_counts.append(100 * wins / total)
            tie_counts.append(100 * ties / total)
            loss_counts.append(100 * losses / total)
            method_labels.append(get_method_label(gpt_method))
    x = np.arange(len(method_labels))
    width = 0.6
    ax.bar(x, win_counts, width, label="GPT Wins", color="#009E73")
    ax.bar(x, tie_counts, width, bottom=win_counts, label="Ties", color="#E69F00")
    ax.bar(x, loss_counts, width, bottom=[w + t for w, t in zip(win_counts, tie_counts)], label="VBC Wins", color="#D55E00")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("GPT Method")
    ax.set_title("Win/Tie/Loss Rate vs VBC (All Topics)")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    for i in range(len(method_labels)):
        ax.annotate(f"{win_counts[i]:.1f}%", xy=(i, win_counts[i]/2), ha="center", va="center", fontsize=9, color="white")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_zero_breakdown(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    zero_pcts, nonzero_pcts, method_labels, colors = [], [], [], []
    for method in METHOD_ORDER:
        method_data = df[df["method"] == method]["epsilon"].dropna().values
        if len(method_data) == 0:
            continue
        pct_zero = 100 * np.sum(method_data == 0) / len(method_data)
        pct_nonzero = 100 - pct_zero
        zero_pcts.append(pct_zero)
        nonzero_pcts.append(pct_nonzero)
        method_labels.append(get_method_label(method))
        colors.append(get_method_color(method))
    x = np.arange(len(method_labels))
    width = 0.6
    ax.bar(x, zero_pcts, width, label="epsilon = 0", color="#009E73", alpha=0.8)
    ax.bar(x, nonzero_pcts, width, bottom=zero_pcts, label="epsilon > 0", color="#D55E00", alpha=0.8)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Method")
    ax.set_title("Zero vs Non-Zero Epsilon by Method (All Topics)")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    for i in range(len(method_labels)):
        ax.annotate(f"{zero_pcts[i]:.1f}%", xy=(i, zero_pcts[i]/2), ha="center", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_cdf_nonzero(df, topic, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna() & (df["epsilon"] > 0)]
    if topic_df.empty:
        plt.close()
        return
    for method in METHOD_ORDER:
        method_data = topic_df[topic_df["method"] == method]["epsilon"].values
        if len(method_data) == 0:
            continue
        sorted_data = np.sort(method_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        lw = 3 if method == "veto_by_consumption" else 2
        ls = "--" if method == "random" else "-"  # Dashed line for Random baseline
        ax.step(sorted_data, cdf, where="post", label=f"{get_method_label(method)} (n={len(method_data)})", color=get_method_color(method), linewidth=lw, linestyle=ls)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Critical Epsilon (non-zero only)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: CDF of Non-Zero Epsilon")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_contingency_table(df, output_path):
    gpt_methods = [m for m in METHOD_ORDER if m not in ["veto_by_consumption", "random_insertion", "random"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for idx, gpt_method in enumerate(gpt_methods):
        ax = axes[idx]
        both_zero, vbc_zero_gpt_nonzero, vbc_nonzero_gpt_zero, both_nonzero = 0, 0, 0, 0
        for topic in ALL_TOPICS:
            topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
            vbc_df = topic_df[topic_df["method"] == "veto_by_consumption"].set_index(["rep_id", "mini_rep_id"])
            gpt_df = topic_df[topic_df["method"] == gpt_method].set_index(["rep_id", "mini_rep_id"])
            common_idx = vbc_df.index.intersection(gpt_df.index)
            if len(common_idx) == 0:
                continue
            vbc_eps = vbc_df.loc[common_idx, "epsilon"].values
            gpt_eps = gpt_df.loc[common_idx, "epsilon"].values
            both_zero += np.sum((vbc_eps == 0) & (gpt_eps == 0))
            vbc_zero_gpt_nonzero += np.sum((vbc_eps == 0) & (gpt_eps > 0))
            vbc_nonzero_gpt_zero += np.sum((vbc_eps > 0) & (gpt_eps == 0))
            both_nonzero += np.sum((vbc_eps > 0) & (gpt_eps > 0))
        matrix = np.array([[both_zero, vbc_zero_gpt_nonzero], [vbc_nonzero_gpt_zero, both_nonzero]])
        total = matrix.sum()
        pct_matrix = 100 * matrix / total if total > 0 else matrix
        sns.heatmap(pct_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=["GPT e=0", "GPT e>0"], yticklabels=["VBC e=0", "VBC e>0"], ax=ax, cbar=False)
        ax.set_title(f"{get_method_label(gpt_method)} vs VBC\n(% of {total} pairs)")
    fig.suptitle("Contingency Tables: Zero vs Non-Zero Epsilon", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_epsilon_when_vbc_fails(df, output_path):
    gpt_methods = [m for m in METHOD_ORDER if m not in ["veto_by_consumption", "random_insertion", "random"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for gpt_method in gpt_methods:
        gpt_eps_when_vbc_fails = []
        for topic in ALL_TOPICS:
            topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
            vbc_df = topic_df[topic_df["method"] == "veto_by_consumption"].set_index(["rep_id", "mini_rep_id"])
            gpt_df = topic_df[topic_df["method"] == gpt_method].set_index(["rep_id", "mini_rep_id"])
            common_idx = vbc_df.index.intersection(gpt_df.index)
            if len(common_idx) == 0:
                continue
            vbc_eps = vbc_df.loc[common_idx, "epsilon"].values
            gpt_eps = gpt_df.loc[common_idx, "epsilon"].values
            mask = vbc_eps > 0
            gpt_eps_when_vbc_fails.extend(gpt_eps[mask])
        if len(gpt_eps_when_vbc_fails) > 0:
            sorted_data = np.sort(gpt_eps_when_vbc_fails)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.step(sorted_data, cdf, where="post", label=f"{get_method_label(gpt_method)} (n={len(gpt_eps_when_vbc_fails)})", color=get_method_color(gpt_method), linewidth=2)
    ax.set_xlabel("GPT Epsilon (when VBC e > 0)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("GPT Method Performance When VBC Fails (e > 0)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_strip_plot(df, topic, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
    if topic_df.empty:
        plt.close()
        return
    plot_df = topic_df.copy()
    plot_df["Method"] = plot_df["method"].map(get_method_label)
    method_order = [get_method_label(m) for m in METHOD_ORDER if m in plot_df["method"].values]
    ax1 = axes[0]
    zero_df = plot_df[plot_df["epsilon"] == 0]
    if not zero_df.empty:
        counts = zero_df.groupby("Method").size().reindex(method_order, fill_value=0)
        ax1.bar(range(len(counts)), counts.values, color=[get_method_color(m) for m in METHOD_ORDER if get_method_label(m) in method_order], alpha=0.7)
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels(counts.index, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Cases with e = 0")
    ax2 = axes[1]
    nonzero_df = plot_df[plot_df["epsilon"] > 0]
    if not nonzero_df.empty:
        palette = {get_method_label(m): get_method_color(m) for m in METHOD_ORDER}
        sns.stripplot(data=nonzero_df, x="Method", y="epsilon", order=method_order, palette=palette, alpha=0.5, jitter=True, ax=ax2)
        ax2.set_xlabel("Method")
        ax2.set_ylabel("Epsilon")
        ax2.set_title("Cases with e > 0")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(f"{TOPIC_DISPLAY_NAMES.get(topic, topic.title())}: Zero vs Non-Zero Epsilon", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_threshold_exceedance(df, output_path, thresholds=None):
    if thresholds is None:
        thresholds = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        method_data = df[df["method"] == method]["epsilon"].dropna().values
        if len(method_data) == 0:
            continue
        exceedance_probs = [100 * np.mean(method_data > t) for t in thresholds]
        lw = 3 if method == "veto_by_consumption" else 2
        ls = "--" if method == "random" else "-"  # Dashed line for Random baseline
        ax.plot(thresholds, exceedance_probs, label=get_method_label(method), color=get_method_color(method), linewidth=lw, linestyle=ls, marker="o", markersize=4)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("P(e > threshold) %")
    ax.set_title("Threshold Exceedance Probability (All Topics)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, max(thresholds) + 0.01)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def generate_extended_stats_table(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHOD_ORDER:
        method_data = df[df["method"] == method]["epsilon"].dropna().values
        if len(method_data) == 0:
            continue
        mean = np.mean(method_data)
        std = np.std(method_data)
        median = np.median(method_data)
        ci_low, ci_high = compute_ci(method_data)
        p10 = np.percentile(method_data, 10)
        p25 = np.percentile(method_data, 25)
        p75 = np.percentile(method_data, 75)
        p90 = np.percentile(method_data, 90)
        n_zero = np.sum(method_data == 0)
        pct_zero = 100 * n_zero / len(method_data)
        pct_gt_01 = 100 * np.mean(method_data > 0.01)
        pct_gt_05 = 100 * np.mean(method_data > 0.05)
        pct_gt_10 = 100 * np.mean(method_data > 0.10)
        min_val = np.min(method_data)
        max_val = np.max(method_data)
        rows.append({"Method": get_method_label(method), "N": len(method_data), "Mean": mean, "Std": std, "Median": median, "95% CI Low": ci_low, "95% CI High": ci_high, "P10": p10, "P25": p25, "P75": p75, "P90": p90, "Min": min_val, "Max": max_val, "% Zero": pct_zero, "% > 0.01": pct_gt_01, "% > 0.05": pct_gt_05, "% > 0.10": pct_gt_10})
    extended_df = pd.DataFrame(rows)
    csv_path = output_dir / "extended_statistics.csv"
    extended_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")
    latex_path = output_dir / "extended_statistics.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Extended Summary Statistics by Method}\n\\label{tab:extended_stats}\n\\scriptsize\n\\begin{tabular}{lccccccccc}\n\\toprule\nMethod & N & Mean & Median & P25 & P75 & \\% Zero & \\% $>$0.01 & \\% $>$0.05 & \\% $>$0.10 \\\\\n\\midrule\n")
        for _, row in extended_df.iterrows():
            f.write(f"{row['Method']} & {row['N']} & {row['Mean']:.3f} & {row['Median']:.3f} & {row['P25']:.3f} & {row['P75']:.3f} & {row['% Zero']:.1f} & {row['% > 0.01']:.1f} & {row['% > 0.05']:.1f} & {row['% > 0.10']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {latex_path}")
    vbc_data = df[df["method"] == "veto_by_consumption"]["epsilon"].dropna().values
    if len(vbc_data) > 0:
        comparison_rows = []
        for gpt_method in [m for m in METHOD_ORDER if m not in ["veto_by_consumption", "random"]]:
            gpt_data = df[df["method"] == gpt_method]["epsilon"].dropna().values
            if len(gpt_data) == 0:
                continue
            gpt_wins, vbc_wins, ties = 0, 0, 0
            for topic in ALL_TOPICS:
                topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
                vbc_df = topic_df[topic_df["method"] == "veto_by_consumption"].set_index(["rep_id", "mini_rep_id"])
                gpt_df = topic_df[topic_df["method"] == gpt_method].set_index(["rep_id", "mini_rep_id"])
                common_idx = vbc_df.index.intersection(gpt_df.index)
                if len(common_idx) == 0:
                    continue
                vbc_eps = vbc_df.loc[common_idx, "epsilon"].values
                gpt_eps = gpt_df.loc[common_idx, "epsilon"].values
                gpt_wins += np.sum(gpt_eps < vbc_eps)
                vbc_wins += np.sum(gpt_eps > vbc_eps)
                ties += np.sum(gpt_eps == vbc_eps)
            total = gpt_wins + vbc_wins + ties
            stat, pval = stats.mannwhitneyu(gpt_data, vbc_data, alternative="two-sided")
            comparison_rows.append({"Method": get_method_label(gpt_method), "N_Pairs": total, "GPT Wins": gpt_wins, "VBC Wins": vbc_wins, "Ties": ties, "GPT Win %": 100 * gpt_wins / total if total > 0 else 0, "Mann-Whitney U": stat, "p-value": pval})
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = output_dir / "vbc_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, float_format="%.4f")
        print(f"Saved: {comparison_path}")

def generate_custom_stats_tables(df, output_dir):
    """Generate simplified stats tables with custom columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_stats(data):
        if len(data) == 0:
            return {}
        return {
            "Mean": np.mean(data),
            "P90": np.percentile(data, 90),
            "P95": np.percentile(data, 95),
            "P99": np.percentile(data, 99),
            "% Zero": 100 * np.mean(data == 0),
            "% <0.01": 100 * np.mean(data < 0.01),
            "% <0.05": 100 * np.mean(data < 0.05),
            "% <0.1": 100 * np.mean(data < 0.1),
        }
    
    # Overall stats
    rows = []
    for method in METHOD_ORDER:
        method_data = df[df["method"] == method]["epsilon"].dropna().values
        if len(method_data) > 0:
            row = {"Method": get_method_label(method)}
            row.update(compute_stats(method_data))
            rows.append(row)
    overall_path = output_dir / "stats_overall.csv"
    pd.DataFrame(rows).to_csv(overall_path, index=False, float_format="%.4f")
    print(f"Saved: {overall_path}")
    
    # Per-topic stats
    for topic in ALL_TOPICS:
        topic_df = df[df["topic"] == topic]
        rows = []
        for method in METHOD_ORDER:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].dropna().values
            if len(method_data) > 0:
                row = {"Method": get_method_label(method)}
                row.update(compute_stats(method_data))
                rows.append(row)
        topic_path = output_dir / f"stats_{topic}.csv"
        pd.DataFrame(rows).to_csv(topic_path, index=False, float_format="%.4f")
        print(f"Saved: {topic_path}")
    
    # Top 3 most controversial topics (abortion, electoral, healthcare)
    top3_topics = ["abortion", "electoral", "healthcare"]
    top3_df = df[df["topic"].isin(top3_topics)]
    rows = []
    for method in METHOD_ORDER:
        method_data = top3_df[top3_df["method"] == method]["epsilon"].dropna().values
        if len(method_data) > 0:
            row = {"Method": get_method_label(method)}
            row.update(compute_stats(method_data))
            rows.append(row)
    top3_path = output_dir / "stats_top3_controversial.csv"
    pd.DataFrame(rows).to_csv(top3_path, index=False, float_format="%.4f")
    print(f"Saved: {top3_path}")
    
    # Bottom 3 least controversial topics (trust, environment, policing)
    bottom3_topics = ["trust", "environment", "policing"]
    bottom3_df = df[df["topic"].isin(bottom3_topics)]
    rows = []
    for method in METHOD_ORDER:
        method_data = bottom3_df[bottom3_df["method"] == method]["epsilon"].dropna().values
        if len(method_data) > 0:
            row = {"Method": get_method_label(method)}
            row.update(compute_stats(method_data))
            rows.append(row)
    bottom3_path = output_dir / "stats_bottom3_less_controversial.csv"
    pd.DataFrame(rows).to_csv(bottom3_path, index=False, float_format="%.4f")
    print(f"Saved: {bottom3_path}")

def generate_traditional_pivot_tables(df, output_dir):
    """Generate pivot tables for traditional methods with topics as columns."""
    trad_dir = output_dir / "traditional"
    trad_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full data including traditional methods (not filtered by METHODS_TO_COMPARE)
    full_df = collect_all_results()
    full_df = full_df[(full_df["voter_dist"] == "uniform") & (full_df["alt_dist"] == "persona_no_context")].copy()
    
    # Add random baseline
    random_df = load_random_baseline()
    if not random_df.empty:
        full_df = pd.concat([full_df, random_df], ignore_index=True)
    
    traditional_methods = ["veto_by_consumption", "borda", "schulze", "irv", "random", "plurality"]
    topic_cols = [TOPIC_DISPLAY_NAMES.get(t, t) for t in ALL_TOPICS]
    
    # Add labels for traditional methods
    trad_labels = {
        "veto_by_consumption": "VBC",
        "borda": "Borda",
        "schulze": "Schulze",
        "irv": "IRV",
        "plurality": "Plurality",
        "random": "Random",
    }
    
    def write_latex_with_bold_best(pivot_df, path, value_cols, higher_is_better=False):
        """Write LaTeX table with bold best values per column."""
        with open(path, "w") as f:
            f.write("\\begin{tabular}{l" + "c" * len(value_cols) + "}\n\\toprule\n")
            f.write(" & ".join(pivot_df.columns) + " \\\\\n\\midrule\n")
            best = {col: (pivot_df[col].max() if higher_is_better else pivot_df[col].min()) for col in value_cols}
            for _, row in pivot_df.iterrows():
                cells = []
                for col in pivot_df.columns:
                    val = row[col]
                    if col in value_cols and pd.notna(val):
                        cells.append(f"\\textbf{{{val:.4f}}}" if val == best[col] else f"{val:.4f}")
                    else:
                        cells.append(str(val))
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    
    def compute_pivot(stat_fn):
        rows = []
        for method in traditional_methods:
            row = {"Method": trad_labels.get(method, method)}
            for topic in ALL_TOPICS:
                data = full_df[(full_df["method"] == method) & (full_df["topic"] == topic)]["epsilon"].dropna().values
                row[TOPIC_DISPLAY_NAMES.get(topic, topic)] = stat_fn(data) if len(data) > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
    
    # Metrics where lower is better
    lower_better = [
        ("mean", lambda d: np.mean(d)),
        ("p90", lambda d: np.percentile(d, 90)),
        ("p95", lambda d: np.percentile(d, 95)),
        ("p99", lambda d: np.percentile(d, 99)),
    ]
    
    # Metrics where higher is better (more zeros/low values = good)
    higher_better = [
        ("pct_zero", lambda d: 100 * np.mean(d == 0)),
        ("pct_lt_0.01", lambda d: 100 * np.mean(d < 0.01)),
        ("pct_lt_0.05", lambda d: 100 * np.mean(d < 0.05)),
        ("pct_lt_0.1", lambda d: 100 * np.mean(d < 0.1)),
    ]
    
    for name, fn in lower_better:
        pivot_df = compute_pivot(fn)
        pivot_df.to_csv(trad_dir / f"{name}_by_topic.csv", index=False, float_format="%.4f")
        write_latex_with_bold_best(pivot_df, trad_dir / f"{name}_by_topic.tex", topic_cols, higher_is_better=False)
        print(f"Saved: {trad_dir}/{name}_by_topic.csv/.tex")
    
    for name, fn in higher_better:
        pivot_df = compute_pivot(fn)
        pivot_df.to_csv(trad_dir / f"{name}_by_topic.csv", index=False, float_format="%.4f")
        write_latex_with_bold_best(pivot_df, trad_dir / f"{name}_by_topic.tex", topic_cols, higher_is_better=True)
        print(f"Saved: {trad_dir}/{name}_by_topic.csv/.tex")

def main():
    print("=" * 60)
    print("GENERATING PAPER-QUALITY PLOTS")
    print("=" * 60)
    df = load_and_filter_data()
    if df.empty:
        print("ERROR: No data found after filtering!")
        return
    PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Generating Cross-Topic Summaries ---")
    generate_summary_table(df, PAPER_OUTPUT_DIR / "tables")
    generate_extended_stats_table(df, PAPER_OUTPUT_DIR / "tables")
    generate_custom_stats_tables(df, PAPER_OUTPUT_DIR / "tables")
    tables_dir = project_root / "outputs" / "paper" / "tables"
    generate_traditional_pivot_tables(df, tables_dir)
    plot_heatmap_method_topic(df, PAPER_OUTPUT_DIR / "heatmap_method_topic.png")
    plot_win_tie_loss(df, PAPER_OUTPUT_DIR / "win_tie_loss.png")
    plot_zero_breakdown(df, PAPER_OUTPUT_DIR / "zero_breakdown.png")
    plot_contingency_table(df, PAPER_OUTPUT_DIR / "contingency_tables.png")
    plot_epsilon_when_vbc_fails(df, PAPER_OUTPUT_DIR / "epsilon_when_vbc_fails.png")
    plot_threshold_exceedance(df, PAPER_OUTPUT_DIR / "threshold_exceedance.png")
    print("\n--- Generating Per-Topic Plots ---")
    for topic in ALL_TOPICS:
        print(f"\n  Topic: {topic}")
        topic_dir = PAPER_OUTPUT_DIR / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        plot_cdf_vbc_vs_gpt(df, topic, topic_dir / f"cdf_{topic}.png", x_max=0.5, y_min=0.5, zoomed=True)
        plot_cdf_vbc_vs_gpt(df, topic, topic_dir / f"cdf_{topic}_full.png", x_max=1.0, y_min=0.0, zoomed=False)
        plot_boxplot(df, topic, topic_dir / f"boxplot_{topic}.png")
        plot_scatter_vbc_vs_gpt(df, topic, topic_dir / f"scatter_{topic}.png")
        plot_bar_epsilon(df, topic, topic_dir / f"bar_ci_{topic}.png")
        plot_cdf_nonzero(df, topic, topic_dir / f"cdf_nonzero_{topic}.png")
        plot_strip_plot(df, topic, topic_dir / f"strip_{topic}.png")
    print("\n--- Generating Combined Multi-Topic CDF ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, topic in enumerate(ALL_TOPICS):
        ax = axes[idx]
        topic_df = df[(df["topic"] == topic) & df["epsilon"].notna()]
        if topic_df.empty:
            continue
        for method in METHOD_ORDER:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            lw = 2.5 if method == "veto_by_consumption" else 1.5
            ls = "--" if method == "random" else "-"  # Dashed line for Random baseline
            ax.step(sorted_data, cdf, where="post", label=get_method_label(method), color=get_method_color(method), linewidth=lw, linestyle=ls)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0.5, 1.02)
        ax.set_xlabel("Critical Epsilon")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(TOPIC_DISPLAY_NAMES.get(topic, topic.title()))
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("CDF Comparison: VBC vs GPT Methods by Topic", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    combined_path = PAPER_OUTPUT_DIR / "combined_cdf_all_topics.png"
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved: {combined_path}")
    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE")
    print(f"Output directory: {PAPER_OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
