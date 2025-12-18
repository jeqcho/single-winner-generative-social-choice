"""
Model Comparison Experiment

Compare preference rankings across different GPT-5 model and reasoning configurations,
measuring Kendall-tau distance and Kendall tau-b relative to gpt-5.2-none.

Models tested:
- gpt-5-nano with reasoning=None
- gpt-5-nano with reasoning=low
- gpt-5-mini with reasoning=None
- gpt-5-mini with reasoning=low
- gpt-5.2 with reasoning=None (reference)

Outputs (to outputs/check-models/):
- rankings.json: All persona rankings for all models
- Kendall-tau distance: results CSV, variance CSV, LaTeX table, bar chart
- Kendall tau-b: results CSV, variance CSV, LaTeX table, bar chart
"""

import csv
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from functools import cmp_to_key
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Constants
PROJECT_ROOT = Path("/home/ec2-user/single-winner-generative-social-choice")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "check-models"
STATEMENTS_FILE = PROJECT_ROOT / "data" / "large_scale" / "prod" / "statements" / "what-should-guide-laws-concerning-abortion.json"
PERSONAS_FILE = PROJECT_ROOT / "data" / "personas" / "prod" / "discriminative.json"

NUM_STATEMENTS = 100
NUM_PERSONAS = 50
SEED = 42
MAX_WORKERS = 50
TOPIC = "What should guide laws concerning abortion?"

# Model configurations
MODELS = [
    {"name": "gpt-5-nano", "reasoning": None, "label": "gpt-5-nano-none"},
    {"name": "gpt-5-nano", "reasoning": "low", "label": "gpt-5-nano-low"},
    {"name": "gpt-5-mini", "reasoning": None, "label": "gpt-5-mini-none"},
    {"name": "gpt-5-mini", "reasoning": "low", "label": "gpt-5-mini-low"},
    {"name": "gpt-5.2", "reasoning": None, "label": "gpt-5.2-none"},  # Reference
]

REFERENCE_MODEL_LABEL = "gpt-5.2-none"


def setup_logging():
    """Setup logging to both file and console."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def load_statements(limit: int = NUM_STATEMENTS) -> List[Dict]:
    """Load and randomly sample statements."""
    with open(STATEMENTS_FILE, 'r') as f:
        data = json.load(f)
    
    random.seed(SEED)
    if len(data) > limit:
        sampled = random.sample(data, limit)
    else:
        sampled = data[:limit]
    
    logging.info(f"Loaded {len(sampled)} statements from {STATEMENTS_FILE}")
    return sampled


def load_personas(limit: int = NUM_PERSONAS) -> List[str]:
    """Load and randomly sample personas."""
    with open(PERSONAS_FILE, 'r') as f:
        personas = json.load(f)
    
    random.seed(SEED)
    sampled = random.sample(personas, min(limit, len(personas)))
    logging.info(f"Sampled {len(sampled)} personas with seed={SEED}")
    return sampled


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def pairwise_compare(
    model_config: Dict,
    persona: str,
    stmt_a: Dict,
    stmt_b: Dict,
    topic: str
) -> int:
    """
    Compare two statements using specified model configuration.
    
    Returns:
        -1 if persona prefers A, 1 if persona prefers B, 0 if equal
    """
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Compare these two statements and indicate which one you prefer:

Statement A: {stmt_a['statement']}

Statement B: {stmt_b['statement']}

Return your choice as JSON: {{"preference": "A"}} or {{"preference": "B"}} or {{"preference": "equal"}}
Return only JSON, no other text."""

    # Build API call parameters
    api_params = {
        "model": model_config["name"],
        "input": [
            {"role": "system", "content": "You are evaluating statements. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
    }
    
    # Add reasoning parameter if specified
    if model_config["reasoning"] is not None:
        api_params["reasoning"] = {"effort": model_config["reasoning"]}
    
    response = client.responses.create(**api_params)
    
    result = json.loads(response.output_text)
    pref = result.get("preference", "equal").upper()
    
    if pref == "A":
        return -1
    elif pref == "B":
        return 1
    return 0


def rank_statements_for_persona(
    model_config: Dict,
    persona: str,
    statements: List[Dict],
    topic: str,
    persona_idx: int,
    intermediate_dir: Path
) -> List[int]:
    """
    Rank statements for a single persona with intermediate caching.
    
    Returns:
        List of statement indices in order from most to least preferred
    """
    cache_file = intermediate_dir / f"{model_config['label']}_persona_{persona_idx}.json"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
        logging.info(f"  [{model_config['label']}] Persona {persona_idx}: loaded from cache")
        return data["ranking"]
    
    # Create indexed statements
    indexed = [{"index": i, "statement": stmt["statement"]} for i, stmt in enumerate(statements)]
    
    comparison_count = [0]
    
    def compare(a: Dict, b: Dict) -> int:
        comparison_count[0] += 1
        return pairwise_compare(model_config, persona, a, b, topic)
    
    # Sort using Python's built-in sort
    sorted_stmts = sorted(indexed, key=cmp_to_key(compare))
    ranking = [s["index"] for s in sorted_stmts]
    
    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({
            "model": model_config['label'],
            "persona_idx": persona_idx,
            "ranking": ranking,
            "comparisons": comparison_count[0]
        }, f)
    
    logging.info(f"  [{model_config['label']}] Persona {persona_idx}: {comparison_count[0]} comparisons")
    return ranking


def get_rankings_for_model(
    model_config: Dict,
    personas: List[str],
    statements: List[Dict],
    topic: str,
    intermediate_dir: Path,
    max_workers: int = MAX_WORKERS
) -> List[List[int]]:
    """
    Get preference rankings from all personas for a specific model.
    
    Returns:
        List of rankings, one per persona. Each ranking is a list of statement indices.
    """
    n_personas = len(personas)
    logging.info(f"\n{'='*60}")
    logging.info(f"Getting rankings for model: {model_config['label']}")
    logging.info(f"{'='*60}")
    
    def process_persona(args):
        idx, persona = args
        return idx, rank_statements_for_persona(
            model_config, persona, statements, topic, idx, intermediate_dir
        )
    
    rankings = [None] * n_personas
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_persona, (i, persona)): i
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Ranking [{model_config['label']}]",
            unit="persona"
        ):
            try:
                idx, ranking = future.result()
                rankings[idx] = ranking
            except Exception as e:
                logging.error(f"Error processing persona {futures[future]}: {e}")
                # Fallback to sequential ranking
                rankings[futures[future]] = list(range(len(statements)))
    
    return rankings


def compute_metrics(
    model_rankings: List[List[int]],
    reference_rankings: List[List[int]]
) -> Tuple[List[float], List[float]]:
    """
    Compute Kendall tau-b and tau distance for each persona.
    
    Returns:
        Tuple of (tau_b_values, tau_distance_values) for each persona
    """
    tau_b_values = []
    tau_distance_values = []
    
    for model_ranking, ref_ranking in zip(model_rankings, reference_rankings):
        tau_b, _ = kendalltau(model_ranking, ref_ranking)
        tau_distance = (1 - tau_b) / 2  # Normalized to [0, 1]
        
        tau_b_values.append(tau_b)
        tau_distance_values.append(tau_distance)
    
    return tau_b_values, tau_distance_values


def save_all_rankings(all_rankings: Dict[str, List[List[int]]], output_path: Path):
    """Save all persona rankings for all models to JSON."""
    with open(output_path, 'w') as f:
        json.dump(all_rankings, f, indent=2)
    logging.info(f"Saved all rankings to {output_path}")


def write_results_csv(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Write main results CSV with mean and std."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", f"Mean {metric_name}", f"Std {metric_name}"])
        
        for label, data in results.items():
            writer.writerow([label, f"{data['mean']:.6f}", f"{data['std']:.6f}"])
    
    logging.info(f"Saved {metric_name} results to {output_path}")


def write_variance_csv(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Write per-persona values CSV."""
    # Get number of personas from first result
    first_key = list(results.keys())[0]
    num_personas = len(results[first_key]['values'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["Model"] + [f"Persona_{i}" for i in range(num_personas)]
        writer.writerow(header)
        
        for label, data in results.items():
            row = [label] + [f"{v:.6f}" for v in data['values']]
            writer.writerow(row)
    
    logging.info(f"Saved {metric_name} per-persona values to {output_path}")


def write_latex_table(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Write LaTeX table with mean ± std format."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{metric_name} by Model (relative to {REFERENCE_MODEL_LABEL})}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        f"Model & {metric_name} \\\\",
        r"\midrule",
    ]
    
    for label, data in results.items():
        mean = data['mean']
        std = data['std']
        lines.append(f"{label.replace('_', '-')} & ${mean:.4f} \\pm {std:.4f}$ \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Saved {metric_name} LaTeX table to {output_path}")


def create_bar_chart(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Create bar chart with error bars."""
    labels = list(results.keys())
    means = [results[label]['mean'] for label in labels]
    stds = [results[label]['std'] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with error bars
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    
    # Highlight reference model
    ref_idx = labels.index(REFERENCE_MODEL_LABEL) if REFERENCE_MODEL_LABEL in labels else -1
    if ref_idx >= 0:
        bars[ref_idx].set_color('forestgreen')
    
    # Labels and title
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} by Model (relative to {REFERENCE_MODEL_LABEL})', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label.replace('_', '\n') for label in labels], fontsize=10)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logging.info(f"Saved {metric_name} bar chart to {output_path}")


def run_experiment():
    """Main experiment runner."""
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("MODEL COMPARISON EXPERIMENT")
    logging.info(f"Statements: {NUM_STATEMENTS}, Personas: {NUM_PERSONAS}, Seed: {SEED}")
    logging.info(f"Models: {[m['label'] for m in MODELS]}")
    logging.info(f"Reference model: {REFERENCE_MODEL_LABEL}")
    logging.info("=" * 80)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intermediate_dir = OUTPUT_DIR / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("\n📊 Loading data...")
    statements = load_statements()
    personas = load_personas()
    
    # Get rankings for each model
    all_rankings: Dict[str, List[List[int]]] = {}
    
    for model_config in MODELS:
        rankings = get_rankings_for_model(
            model_config, personas, statements, TOPIC, intermediate_dir
        )
        all_rankings[model_config['label']] = rankings
    
    # Save all rankings
    save_all_rankings(all_rankings, OUTPUT_DIR / "rankings.json")
    
    # Get reference rankings
    reference_rankings = all_rankings[REFERENCE_MODEL_LABEL]
    
    # Compute metrics for each model
    tau_b_results: Dict[str, Dict] = {}
    tau_distance_results: Dict[str, Dict] = {}
    
    logging.info("\n📈 Computing metrics...")
    
    for model_label, model_rankings in all_rankings.items():
        tau_b_values, tau_distance_values = compute_metrics(model_rankings, reference_rankings)
        
        tau_b_results[model_label] = {
            'values': tau_b_values,
            'mean': np.mean(tau_b_values),
            'std': np.std(tau_b_values)
        }
        
        tau_distance_results[model_label] = {
            'values': tau_distance_values,
            'mean': np.mean(tau_distance_values),
            'std': np.std(tau_distance_values)
        }
        
        logging.info(f"  {model_label}: tau-b = {tau_b_results[model_label]['mean']:.4f} ± {tau_b_results[model_label]['std']:.4f}, "
                     f"tau-distance = {tau_distance_results[model_label]['mean']:.4f} ± {tau_distance_results[model_label]['std']:.4f}")
    
    # Generate outputs for Kendall-tau distance
    logging.info("\n📝 Generating Kendall-tau distance outputs...")
    write_results_csv(tau_distance_results, OUTPUT_DIR / "kendall_tau_results.csv", "Kendall-Tau Distance")
    write_variance_csv(tau_distance_results, OUTPUT_DIR / "kendall_tau_variance.csv", "Kendall-Tau Distance")
    write_latex_table(tau_distance_results, OUTPUT_DIR / "kendall_tau_table.tex", "Kendall-Tau Distance")
    create_bar_chart(tau_distance_results, OUTPUT_DIR / "kendall_tau_chart.png", "Kendall-Tau Distance")
    
    # Generate outputs for Kendall tau-b
    logging.info("\n📝 Generating Kendall tau-b outputs...")
    write_results_csv(tau_b_results, OUTPUT_DIR / "kendall_tau_b_results.csv", "Kendall Tau-b")
    write_variance_csv(tau_b_results, OUTPUT_DIR / "kendall_tau_b_variance.csv", "Kendall Tau-b")
    write_latex_table(tau_b_results, OUTPUT_DIR / "kendall_tau_b_table.tex", "Kendall Tau-b")
    create_bar_chart(tau_b_results, OUTPUT_DIR / "kendall_tau_b_chart.png", "Kendall Tau-b")
    
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT COMPLETED")
    logging.info(f"All outputs saved to: {OUTPUT_DIR}")
    logging.info("=" * 80)
    
    return all_rankings, tau_b_results, tau_distance_results


if __name__ == "__main__":
    run_experiment()


