"""
Persona Embeddings and Clustering Analysis.

This script:
1. Loads all 1000 personas from data/personas/prod/full.json
2. Generates embeddings using OpenAI text-embedding-3-small
3. Runs k-means clustering for k=5, 10, 20
4. Generates PCA visualizations
5. Generates cluster descriptions via GPT-5.2
6. Creates a markdown report with tables
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PERSONAS_FILE = DATA_DIR / "personas" / "prod" / "full.json"
EMBEDDINGS_BASE_DIR = DATA_DIR / "persona_embeddings"
REPORTS_BASE_DIR = PROJECT_ROOT / "reports" / "persona_clustering"

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
GPT_MODEL = "gpt-5.2"
DEFAULT_K_VALUES = [5, 10, 20]
BATCH_SIZE = 100  # Batch size for embedding API calls
MAX_CONTEXT_TOKENS = 120000  # Conservative limit for GPT-5.2 context


def load_personas(personas_file: Path) -> List[str]:
    """Load all personas from the specified JSON file."""
    logger.info(f"Loading personas from {personas_file}")
    with open(personas_file, 'r') as f:
        personas = json.load(f)
    logger.info(f"Loaded {len(personas)} personas")
    return personas


def generate_embeddings(
    personas: List[str],
    client: OpenAI,
    embeddings_dir: Path,
    force_regenerate: bool = False
) -> np.ndarray:
    """
    Generate embeddings for all personas using OpenAI API.
    
    Caches embeddings to avoid re-computation.
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = embeddings_dir / "persona_embeddings.npy"
    
    if embeddings_file.exists() and not force_regenerate:
        logger.info(f"Loading cached embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        if len(embeddings) == len(personas):
            return embeddings
        logger.warning("Cached embeddings size mismatch, regenerating...")
    
    logger.info(f"Generating embeddings for {len(personas)} personas...")
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(personas), BATCH_SIZE), desc="Embedding batches"):
        batch = personas[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    embeddings = np.array(all_embeddings)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    
    # Cache embeddings
    np.save(embeddings_file, embeddings)
    logger.info(f"Saved embeddings to {embeddings_file}")
    
    return embeddings


def run_kmeans_clustering(
    embeddings: np.ndarray,
    k_values: List[int]
) -> Dict[int, np.ndarray]:
    """Run k-means clustering for multiple k values."""
    logger.info(f"Running k-means clustering for k={k_values}")
    
    cluster_assignments = {}
    for k in k_values:
        logger.info(f"Clustering with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        cluster_assignments[k] = labels
        
        # Log cluster sizes
        sizes = [np.sum(labels == i) for i in range(k)]
        logger.info(f"  k={k} cluster sizes: {sizes}")
    
    return cluster_assignments


def generate_visualizations(
    embeddings: np.ndarray,
    cluster_assignments: Dict[int, np.ndarray],
    output_dir: Path
) -> Dict[int, str]:
    """Generate PCA visualizations for each k value."""
    logger.info("Generating PCA visualizations...")
    
    # Fit PCA once
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    
    viz_paths = {}
    for k, labels in cluster_assignments.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a colormap with enough distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, k))
        
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(
                coords[mask, 0], 
                coords[mask, 1], 
                c=[colors[cluster_id]], 
                label=f"Cluster {cluster_id} (n={np.sum(mask)})",
                alpha=0.6,
                s=30
            )
        
        ax.set_xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
        ax.set_title(f"Persona Clusters (k={k})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        viz_path = output_dir / f"cluster_viz_k{k}.png"
        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        viz_paths[k] = viz_path.name
        logger.info(f"Saved visualization to {viz_path}")
    
    return viz_paths


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens in text (1 token ≈ 4 chars)."""
    return len(text) // 4


def generate_cluster_description(
    cluster_personas: List[str],
    cluster_id: int,
    k: int,
    client: OpenAI
) -> str:
    """Generate a description for a cluster using GPT-5.2."""
    
    # Format personas
    formatted_personas = "\n\n---\n\n".join(
        f"Persona {i+1}:\n{p}" for i, p in enumerate(cluster_personas)
    )
    
    # Check if we need to sample due to context length
    total_tokens = estimate_tokens(formatted_personas)
    
    if total_tokens > MAX_CONTEXT_TOKENS:
        # Sample personas to fit within context
        max_personas = int(len(cluster_personas) * (MAX_CONTEXT_TOKENS / total_tokens))
        max_personas = max(10, max_personas)  # At least 10 personas
        
        logger.warning(
            f"Cluster {cluster_id} (k={k}) has {len(cluster_personas)} personas "
            f"(~{total_tokens} tokens), sampling {max_personas} personas"
        )
        
        sampled_personas = random.sample(cluster_personas, max_personas)
        formatted_personas = "\n\n---\n\n".join(
            f"Persona {i+1}:\n{p}" for i, p in enumerate(sampled_personas)
        )
        note = f" (sampled {max_personas} of {len(cluster_personas)} personas)"
    else:
        note = ""
    
    prompt = f"""Here are {len(cluster_personas)} personas from a cluster{note}. 
Describe in 1-2 sentences what characterizes this group of people. Focus on the most distinctive demographic, political, and lifestyle patterns that define this cluster.

{formatted_personas}

Description:"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=200,
    )
    
    return response.choices[0].message.content.strip()


def get_representative_persona(
    cluster_personas: List[str],
    cluster_indices: List[int],
    embeddings: np.ndarray
) -> tuple[str, int]:
    """
    Get the most representative persona (closest to cluster centroid).
    
    Returns the persona text and its original index.
    """
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)
    
    # Find persona closest to centroid
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_idx = np.argmin(distances)
    
    return cluster_personas[closest_idx], cluster_indices[closest_idx]


def truncate_persona_for_table(persona: str, max_chars: int = 200) -> str:
    """Truncate persona for table display, keeping key info."""
    # Extract key fields
    lines = persona.strip().split('\n')
    key_fields = ['age:', 'sex:', 'race:', 'occupation category:', 'political views:', 'religion:']
    
    extracted = []
    for line in lines:
        for field in key_fields:
            if line.lower().startswith(field):
                extracted.append(line.strip())
                break
    
    result = ' | '.join(extracted)
    if len(result) > max_chars:
        result = result[:max_chars-3] + "..."
    
    return result


def generate_markdown_report(
    personas: List[str],
    embeddings: np.ndarray,
    cluster_assignments: Dict[int, np.ndarray],
    viz_paths: Dict[int, str],
    client: OpenAI,
    output_dir: Path
) -> Path:
    """Generate the markdown report with tables for each k value."""
    logger.info("Generating markdown report...")
    
    report_lines = [
        "# Persona Clustering Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total personas**: {len(personas)}",
        f"- **Embedding model**: {EMBEDDING_MODEL}",
        f"- **Embedding dimensions**: {EMBEDDING_DIM}",
        f"- **Clustering algorithm**: K-Means",
        f"- **K values tested**: {sorted(cluster_assignments.keys())}",
        "",
    ]
    
    # Process each k value
    for k in sorted(cluster_assignments.keys()):
        labels = cluster_assignments[k]
        
        report_lines.extend([
            f"## K={k} Clusters",
            "",
        ])
        
        # Build table
        table_rows = []
        
        for cluster_id in range(k):
            # Get personas in this cluster
            cluster_indices = np.where(labels == cluster_id)[0].tolist()
            cluster_personas = [personas[i] for i in cluster_indices]
            cluster_size = len(cluster_personas)
            
            logger.info(f"Generating description for k={k}, cluster {cluster_id} ({cluster_size} personas)...")
            
            # Get description from GPT-5.2
            description = generate_cluster_description(
                cluster_personas, cluster_id, k, client
            )
            
            # Get representative persona
            rep_persona, rep_idx = get_representative_persona(
                cluster_personas, cluster_indices, embeddings
            )
            rep_summary = truncate_persona_for_table(rep_persona)
            
            table_rows.append({
                'cluster': cluster_id,
                'size': cluster_size,
                'description': description.replace('|', '/').replace('\n', ' '),
                'representative': rep_summary.replace('|', '/'),
            })
        
        # Write table
        report_lines.extend([
            "| Cluster | Size | Description | Representative Persona |",
            "|---------|------|-------------|------------------------|",
        ])
        
        for row in table_rows:
            report_lines.append(
                f"| {row['cluster']} | {row['size']} | {row['description']} | {row['representative']} |"
            )
        
        report_lines.extend([
            "",
            f"![Cluster Visualization (k={k})]({viz_paths[k]})",
            "",
        ])
    
    # Write report
    report_path = output_dir / "cluster_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved report to {report_path}")
    return report_path


def save_cluster_data(
    cluster_assignments: Dict[int, np.ndarray],
    embeddings_dir: Path
) -> None:
    """Save cluster assignments to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    data = {
        str(k): labels.tolist() 
        for k, labels in cluster_assignments.items()
    }
    
    output_file = embeddings_dir / "persona_clusters.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved cluster assignments to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate persona embeddings and clustering analysis"
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-generate embeddings even if cached"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
        help="Comma-separated k values for clustering (default: 5,10,20)"
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        default=None,
        help="Path to personas JSON file (default: data/personas/prod/full.json)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output directories (e.g., '_adult' creates persona_embeddings_adult/)"
    )
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    
    # Determine personas file
    if args.personas_file:
        personas_file = Path(args.personas_file)
        if not personas_file.is_absolute():
            personas_file = PROJECT_ROOT / personas_file
    else:
        personas_file = DEFAULT_PERSONAS_FILE
    
    # Determine output directories with suffix
    suffix = args.output_suffix
    embeddings_dir = EMBEDDINGS_BASE_DIR.parent / f"{EMBEDDINGS_BASE_DIR.name}{suffix}"
    reports_dir = REPORTS_BASE_DIR.parent / f"{REPORTS_BASE_DIR.name}{suffix}"
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create output directories
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load personas
    personas = load_personas(personas_file)
    
    # Generate embeddings
    embeddings = generate_embeddings(personas, client, embeddings_dir, args.force_embed)
    
    # Run clustering
    cluster_assignments = run_kmeans_clustering(embeddings, k_values)
    
    # Save cluster assignments
    save_cluster_data(cluster_assignments, embeddings_dir)
    
    # Generate visualizations
    viz_paths = generate_visualizations(embeddings, cluster_assignments, reports_dir)
    
    # Generate report
    report_path = generate_markdown_report(
        personas, embeddings, cluster_assignments, viz_paths, client, reports_dir
    )
    
    logger.info("=" * 50)
    logger.info("Clustering analysis complete!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Visualizations: {reports_dir}")
    logger.info(f"Embeddings: {embeddings_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
