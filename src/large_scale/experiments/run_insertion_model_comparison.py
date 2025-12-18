"""
Model Comparison Experiment using Hybrid Insertion Ranking.

Compare preference rankings across different GPT-5 model and temperature configurations
using the hybrid insertion-based ranking method.

Models tested:
- gpt-5-nano with temperature=1.0 (2 runs)
- gpt-5-mini with temperature=1.0 (2 runs)
- gpt-5.2 with temperature=1.0 (2 runs)
- gpt-5.2 with temperature=0.0 (1 run)

Configuration:
- 50 statements
- 10 personas
- Hybrid insertion sort (threshold=70)

Outputs (to outputs/check-models-insertion/):
- rankings.json: All persona rankings for all models
- kendall_tau_matrix.csv: Pairwise Kendall-tau distance matrix
- kendall_tau_matrix.tex: LaTeX version
- kendall_tau_matrix.png: Heatmap visualization
- Per-model subdirectories with detailed comparisons
"""

import csv
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import kendalltau
from dotenv import load_dotenv
import openai
from tqdm import tqdm

from src.large_scale.insertion_ranking import rank_statements_hybrid

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Constants
PROJECT_ROOT = Path("/home/ec2-user/single-winner-generative-social-choice")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "check-models-insertion"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
STATEMENTS_FILE = PROJECT_ROOT / "data" / "large_scale" / "prod" / "statements" / "what-should-guide-laws-concerning-abortion.json"
PERSONAS_FILE = PROJECT_ROOT / "data" / "personas" / "prod" / "discriminative.json"

NUM_STATEMENTS = 50
NUM_PERSONAS = 10
SEED = 42
MAX_WORKERS = 10  # Parallel personas per model
TOPIC = "What should guide laws concerning abortion?"

# Model configurations
MODELS = [
    {"name": "gpt-5-nano", "temperature": 1.0, "label": "gpt-5-nano-t1-1"},
    {"name": "gpt-5-nano", "temperature": 1.0, "label": "gpt-5-nano-t1-2"},
    {"name": "gpt-5-mini", "temperature": 1.0, "label": "gpt-5-mini-t1-1"},
    {"name": "gpt-5-mini", "temperature": 1.0, "label": "gpt-5-mini-t1-2"},
    {"name": "gpt-5.2", "temperature": 1.0, "label": "gpt-5.2-t1-1"},
    {"name": "gpt-5.2", "temperature": 1.0, "label": "gpt-5.2-t1-2"},
    {"name": "gpt-5.2", "temperature": 0.0, "label": "gpt-5.2-t0"},
]


def setup_logging() -> Path:
    """Setup logging to both file and console."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override handlers set by imported modules
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


def get_rankings_for_model(
    model_config: Dict,
    personas: List[str],
    statements: List[Dict],
    topic: str
) -> List[List[int]]:
    """
    Get rankings for all personas using hybrid insertion sort.
    
    Uses intermediate caching to support restart.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    model_label = model_config["label"]
    model_name = model_config["name"]
    temperature = model_config["temperature"]
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing model: {model_label}")
    logging.info(f"  Model name: {model_name}, Temperature: {temperature}")
    logging.info(f"{'='*60}")
    
    def process_persona(persona_idx: int, persona: str) -> Tuple[int, List[int]]:
        """Process a single persona with caching."""
        cache_file = INTERMEDIATE_DIR / f"{model_label}_persona_{persona_idx}.json"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logging.info(f"  [{model_label}] Persona {persona_idx}: loaded from cache")
            return persona_idx, data["ranking"]
        
        # Run ranking
        logging.info(f"  [{model_label}] Persona {persona_idx}: starting ranking...")
        ranking = rank_statements_hybrid(
            persona=persona,
            statements=statements,
            topic=topic,
            openai_client=client,
            threshold=70,  # Default threshold
            model_name=model_name,
            temperature=temperature
        )
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                "model_label": model_label,
                "persona_idx": persona_idx,
                "ranking": ranking
            }, f)
        
        logging.info(f"  [{model_label}] Persona {persona_idx}: completed and cached")
        return persona_idx, ranking
    
    # Process personas in parallel
    rankings = [None] * len(personas)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_persona, i, persona): i
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Ranking [{model_label}]", unit="persona"):
            idx, ranking = future.result()
            rankings[idx] = ranking
    
    logging.info(f"  [{model_label}] All {len(personas)} personas completed")
    return rankings


def compute_kendall_tau_matrix(all_rankings: Dict[str, List[List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Kendall-tau distance matrix between all models.
    
    Returns:
        Tuple of (tau_distance_matrix, tau_b_matrix) where each cell [i,j]
        is the mean across all personas of tau_distance(model_i, model_j)
    """
    model_labels = list(all_rankings.keys())
    n_models = len(model_labels)
    n_personas = len(all_rankings[model_labels[0]])
    
    tau_distance_matrix = np.zeros((n_models, n_models))
    tau_b_matrix = np.zeros((n_models, n_models))
    
    for i, model_i in enumerate(model_labels):
        for j, model_j in enumerate(model_labels):
            if i == j:
                tau_distance_matrix[i, j] = 0.0
                tau_b_matrix[i, j] = 1.0
                continue
            
            # Compute tau-b for each persona and average
            tau_b_values = []
            for p in range(n_personas):
                ranking_i = all_rankings[model_i][p]
                ranking_j = all_rankings[model_j][p]
                tau_b, _ = kendalltau(ranking_i, ranking_j)
                tau_b_values.append(tau_b)
            
            mean_tau_b = np.mean(tau_b_values)
            mean_tau_distance = (1 - mean_tau_b) / 2
            
            tau_b_matrix[i, j] = mean_tau_b
            tau_distance_matrix[i, j] = mean_tau_distance
    
    return tau_distance_matrix, tau_b_matrix


def save_tau_matrix_csv(matrix: np.ndarray, model_labels: List[str], output_path: Path, metric_name: str):
    """Save Kendall-tau matrix as CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([""] + model_labels)
        
        # Data rows
        for i, label in enumerate(model_labels):
            row = [label] + [f"{matrix[i, j]:.4f}" for j in range(len(model_labels))]
            writer.writerow(row)
    
    logging.info(f"Saved {metric_name} matrix to {output_path}")


def save_tau_matrix_latex(matrix: np.ndarray, model_labels: List[str], output_path: Path, metric_name: str):
    """Save Kendall-tau matrix as LaTeX table."""
    n = len(model_labels)
    escaped_labels = [l.replace('_', '-').replace('.', '-') for l in model_labels]
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{metric_name} Matrix (Pairwise Model Comparison)}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        " & " + " & ".join(escaped_labels) + r" \\",
        r"\midrule",
    ]
    
    for i, label in enumerate(escaped_labels):
        row_values = []
        for j in range(n):
            if i == j:
                row_values.append("-")
            else:
                row_values.append(f"{matrix[i, j]:.3f}")
        lines.append(label + " & " + " & ".join(row_values) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Saved {metric_name} LaTeX table to {output_path}")


def create_tau_heatmap(matrix: np.ndarray, model_labels: List[str], output_path: Path, metric_name: str):
    """Create heatmap visualization of Kendall-tau matrix."""
    n = len(model_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap
    if "distance" in metric_name.lower():
        cmap = 'RdYlGn_r'  # Red = high distance (bad), Green = low distance (good)
        vmin, vmax = 0, 0.5
    else:
        cmap = 'RdYlGn'  # Green = high tau-b (good), Red = low tau-b (bad)
        vmin, vmax = 0, 1
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va="bottom", fontsize=11)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    # Format labels
    display_labels = [l.replace('-', '\n') for l in model_labels]
    ax.set_xticklabels(display_labels, fontsize=9, ha='center')
    ax.set_yticklabels(display_labels, fontsize=9)
    
    # Add cell values
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
            else:
                text = f"{matrix[i, j]:.3f}"
            
            # Choose text color based on background
            val = matrix[i, j]
            if "distance" in metric_name.lower():
                text_color = 'white' if val > 0.25 else 'black'
            else:
                text_color = 'white' if val < 0.5 else 'black'
            
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)
    
    ax.set_title(f'{metric_name}\n(Mean across {NUM_PERSONAS} personas)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved {metric_name} heatmap to {output_path}")


def generate_per_model_outputs(all_rankings: Dict[str, List[List[int]]]):
    """Generate detailed outputs for each model as reference (like generate_all_model_comparisons.py)."""
    model_labels = list(all_rankings.keys())
    
    for ref_model in model_labels:
        # Create directory for this reference model
        dir_name = ref_model.replace(".", "-")
        model_dir = OUTPUT_DIR / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\nGenerating outputs with reference: {ref_model}")
        
        ref_rankings = all_rankings[ref_model]
        n_personas = len(ref_rankings)
        
        # Compute metrics for all models relative to this reference
        tau_b_results = {}
        tau_distance_results = {}
        
        for model_label, model_rankings in all_rankings.items():
            tau_b_values = []
            tau_distance_values = []
            
            for p in range(n_personas):
                tau_b, _ = kendalltau(model_rankings[p], ref_rankings[p])
                tau_distance = (1 - tau_b) / 2
                tau_b_values.append(tau_b)
                tau_distance_values.append(tau_distance)
            
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
        
        # Save CSV results
        for results, name in [(tau_distance_results, "kendall_tau_distance"), 
                               (tau_b_results, "kendall_tau_b")]:
            csv_path = model_dir / f"{name}_results.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Model", "Mean", "Std"])
                for label, data in results.items():
                    writer.writerow([label, f"{data['mean']:.6f}", f"{data['std']:.6f}"])
        
        # Save per-persona variance CSV
        for results, name in [(tau_distance_results, "kendall_tau_distance"),
                               (tau_b_results, "kendall_tau_b")]:
            var_path = model_dir / f"{name}_variance.csv"
            with open(var_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Model"] + [f"Persona_{i}" for i in range(n_personas)]
                writer.writerow(header)
                for label, data in results.items():
                    row = [label] + [f"{v:.6f}" for v in data['values']]
                    writer.writerow(row)
        
        # Create bar chart
        for results, name, ylabel in [(tau_distance_results, "kendall_tau_distance", "Kendall-Tau Distance"),
                                       (tau_b_results, "kendall_tau_b", "Kendall Tau-b")]:
            labels = list(results.keys())
            means = [results[l]['mean'] for l in labels]
            stds = [results[l]['std'] for l in labels]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
            
            # Highlight reference model
            ref_idx = labels.index(ref_model)
            bars[ref_idx].set_color('forestgreen')
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{ylabel} (relative to {ref_model})', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([l.replace('-', '\n') for l in labels], fontsize=9)
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.annotate(f'{mean:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                           ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(model_dir / f"{name}_chart.png", dpi=150)
            plt.close()
        
        logging.info(f"  Saved outputs to {model_dir}")


def main():
    """Main experiment function."""
    log_file = setup_logging()
    
    logging.info("=" * 80)
    logging.info("INSERTION RANKING MODEL COMPARISON EXPERIMENT")
    logging.info(f"Statements: {NUM_STATEMENTS}, Personas: {NUM_PERSONAS}, Seed: {SEED}")
    logging.info(f"Models: {[m['label'] for m in MODELS]}")
    logging.info("=" * 80)
    
    # Step 1: Load data
    logging.info("\n📊 Step 1: Loading data...")
    statements = load_statements()
    personas = load_personas()
    
    # Step 2: Get rankings for each model
    logging.info("\n🗳️ Step 2: Getting preference rankings for all models...")
    all_rankings: Dict[str, List[List[int]]] = {}
    
    for model_config in MODELS:
        rankings = get_rankings_for_model(model_config, personas, statements, TOPIC)
        all_rankings[model_config["label"]] = rankings
    
    # Step 3: Save all rankings
    logging.info("\n💾 Step 3: Saving rankings...")
    rankings_file = OUTPUT_DIR / "rankings.json"
    with open(rankings_file, 'w') as f:
        json.dump(all_rankings, f, indent=2)
    logging.info(f"Saved rankings to {rankings_file}")
    
    # Step 4: Compute and save Kendall-tau matrices
    logging.info("\n📈 Step 4: Computing Kendall-tau matrices...")
    model_labels = list(all_rankings.keys())
    tau_distance_matrix, tau_b_matrix = compute_kendall_tau_matrix(all_rankings)
    
    # Save matrices
    save_tau_matrix_csv(tau_distance_matrix, model_labels, 
                       OUTPUT_DIR / "kendall_tau_distance_matrix.csv", "Kendall-Tau Distance")
    save_tau_matrix_csv(tau_b_matrix, model_labels,
                       OUTPUT_DIR / "kendall_tau_b_matrix.csv", "Kendall Tau-b")
    
    save_tau_matrix_latex(tau_distance_matrix, model_labels,
                         OUTPUT_DIR / "kendall_tau_distance_matrix.tex", "Kendall-Tau Distance")
    save_tau_matrix_latex(tau_b_matrix, model_labels,
                         OUTPUT_DIR / "kendall_tau_b_matrix.tex", "Kendall Tau-b")
    
    create_tau_heatmap(tau_distance_matrix, model_labels,
                      OUTPUT_DIR / "kendall_tau_distance_matrix.png", "Kendall-Tau Distance")
    create_tau_heatmap(tau_b_matrix, model_labels,
                      OUTPUT_DIR / "kendall_tau_b_matrix.png", "Kendall Tau-b")
    
    # Step 5: Generate per-model detailed outputs
    logging.info("\n📊 Step 5: Generating per-model detailed outputs...")
    generate_per_model_outputs(all_rankings)
    
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT COMPLETE!")
    logging.info(f"All outputs saved to: {OUTPUT_DIR}")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 80)
    
    # Print summary
    print("\n📋 SUMMARY:")
    print(f"   Statements: {NUM_STATEMENTS}")
    print(f"   Personas: {NUM_PERSONAS}")
    print(f"   Models: {len(MODELS)}")
    print(f"\n📁 Output files:")
    print(f"   - {OUTPUT_DIR}/rankings.json")
    print(f"   - {OUTPUT_DIR}/kendall_tau_distance_matrix.{{csv,tex,png}}")
    print(f"   - {OUTPUT_DIR}/kendall_tau_b_matrix.{{csv,tex,png}}")
    print(f"   - Per-model directories with detailed outputs")
    print(f"\n📝 To monitor logs:")
    print(f"   tail -f {log_file}")


if __name__ == "__main__":
    main()

