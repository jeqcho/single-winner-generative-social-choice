"""
Generate multi-persona comparison plots for epsilon and epsilon-100.

This script generates barplots and stripplots comparing results
across different persona counts (5, 10, 20).

Usage:
    uv run python -m src.full_experiment.generate_multi_persona_plots
"""

import logging
from pathlib import Path

from .config import OUTPUT_DIR, ALL_TOPICS
from .visualizer import (
    collect_all_results_for_n_personas,
    collect_all_results_for_n_personas_clustered,
    plot_epsilon_multi_persona_barplot,
    plot_epsilon_multi_persona_stripplot,
    PERSONA_COUNTS,
)
from .epsilon_100_plotter import (
    generate_multi_persona_epsilon_100_plots,
)
from .epsilon_100 import (
    collect_all_epsilon_100_for_n_personas,
    collect_all_epsilon_100_for_n_personas_clustered,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_multi_persona_epsilon_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: list = None,
    ablation: str = "full",
    persona_counts: list = None
) -> None:
    """
    Generate multi-persona comparison plots for epsilon (N personas).
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = all)
        ablation: Ablation type
        persona_counts: List of persona counts to compare
    """
    if persona_counts is None:
        persona_counts = PERSONA_COUNTS
    
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
    
    logger.info(f"Collecting epsilon data for persona counts: {persona_counts}")
    
    # Collect results for each persona count
    flat_results_by_n = {}
    clustered_results_by_n = {}
    
    for n_personas in persona_counts:
        flat_results = collect_all_results_for_n_personas(
            n_personas, output_dir, ablation, topics
        )
        clustered_results = collect_all_results_for_n_personas_clustered(
            n_personas, output_dir, ablation, topics
        )
        
        # Only include if we have data
        total_samples = sum(len(v) for v in flat_results.values())
        if total_samples > 0:
            flat_results_by_n[n_personas] = flat_results
            clustered_results_by_n[n_personas] = clustered_results
            logger.info(f"  {n_personas} personas: {total_samples} epsilon values")
        else:
            logger.warning(f"  {n_personas} personas: no data found")
    
    if not flat_results_by_n:
        logger.warning("No data found for any persona count")
        return
    
    # Generate barplot
    logger.info("Generating epsilon barplot...")
    plot_epsilon_multi_persona_barplot(
        clustered_results_by_n,
        title=f"Average Epsilon by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_multi_persona_barplot.png"
    )
    
    # Generate stripplot
    logger.info("Generating epsilon stripplot...")
    plot_epsilon_multi_persona_stripplot(
        flat_results_by_n,
        title=f"Epsilon Distribution by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_multi_persona_stripplot.png"
    )
    
    logger.info("Epsilon multi-persona plots complete!")


def generate_multi_persona_epsilon_100_plots_wrapper(
    output_dir: Path = OUTPUT_DIR,
    topics: list = None,
    ablation: str = "full",
    persona_counts: list = None
) -> None:
    """
    Generate multi-persona comparison plots for epsilon-100.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = all)
        ablation: Ablation type
        persona_counts: List of persona counts to compare
    """
    from .epsilon_100_plotter import (
        plot_epsilon_100_multi_persona_barplot,
        plot_epsilon_100_multi_persona_stripplot,
    )
    
    if persona_counts is None:
        persona_counts = PERSONA_COUNTS
    
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
    
    logger.info(f"Collecting epsilon-100 data for persona counts: {persona_counts}")
    
    # Collect results for each persona count
    flat_results_by_n = {}
    clustered_results_by_n = {}
    
    for n_personas in persona_counts:
        flat_results = collect_all_epsilon_100_for_n_personas(
            n_personas, output_dir, ablation, topics
        )
        clustered_results = collect_all_epsilon_100_for_n_personas_clustered(
            n_personas, output_dir, ablation, topics
        )
        
        # Only include if we have data
        total_samples = sum(len(v) for v in flat_results.values())
        if total_samples > 0:
            flat_results_by_n[n_personas] = flat_results
            clustered_results_by_n[n_personas] = clustered_results
            logger.info(f"  {n_personas} personas: {total_samples} epsilon-100 values")
        else:
            logger.warning(f"  {n_personas} personas: no data found")
    
    if not flat_results_by_n:
        logger.warning("No data found for any persona count")
        return
    
    # Generate barplot
    logger.info("Generating epsilon-100 barplot...")
    plot_epsilon_100_multi_persona_barplot(
        clustered_results_by_n,
        title=f"Average Epsilon (100 Personas) by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_100_multi_persona_barplot.png"
    )
    
    # Generate stripplot
    logger.info("Generating epsilon-100 stripplot...")
    plot_epsilon_100_multi_persona_stripplot(
        flat_results_by_n,
        title=f"Epsilon (100 Personas) Distribution by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_100_multi_persona_stripplot.png"
    )
    
    logger.info("Epsilon-100 multi-persona plots complete!")


def main():
    from .config import ABLATIONS
    
    output_dir = OUTPUT_DIR
    
    # Get all topics
    data_dir = output_dir / "data"
    if data_dir.exists():
        topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir() and not d.name.startswith("pvc")]
        logger.info(f"Found {len(topics)} topics")
    else:
        topics = ALL_TOPICS
        logger.info("Using default topics list")
    
    # Generate plots for all ablations
    for ablation in ABLATIONS:
        logger.info("=" * 60)
        logger.info(f"Generating multi-persona plots for ablation: {ablation}")
        logger.info("=" * 60)
        
        # Generate epsilon plots (computed with N personas)
        logger.info("Generating epsilon plots (computed with N personas)")
        generate_multi_persona_epsilon_plots(
            output_dir=output_dir,
            topics=topics,
            ablation=ablation,
            persona_counts=[5, 10, 20]
        )
        
        # Generate epsilon-100 plots (computed against 100 personas)
        logger.info("Generating epsilon-100 plots (computed against 100 personas)")
        generate_multi_persona_epsilon_100_plots_wrapper(
            output_dir=output_dir,
            topics=topics,
            ablation=ablation,
            persona_counts=[5, 10, 20]
        )
    
    logger.info("=" * 60)
    logger.info("All multi-persona plots generated for all ablations!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

