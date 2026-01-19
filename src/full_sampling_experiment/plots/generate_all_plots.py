"""
Main script to generate all plots for the full sampling experiment.

Usage:
    uv run python -m src.full_sampling_experiment.plots.generate_all_plots
    
Or run individual modules:
    uv run python -m src.full_sampling_experiment.plots.cdf_all_methods
    uv run python -m src.full_sampling_experiment.plots.cdf_4subplot
    uv run python -m src.full_sampling_experiment.plots.cdf_by_topic
    uv run python -m src.full_sampling_experiment.plots.bar_charts
    uv run python -m src.full_sampling_experiment.plots.likert_plots
    uv run python -m src.full_sampling_experiment.plots.summary_stats
"""
import argparse
from .config import FIGURES_DIR
from .cdf_all_methods import plot_cdf_all_methods
from .cdf_4subplot import generate_cdf_4subplot_plots
from .cdf_by_topic import generate_cdf_by_topic_plots
from .bar_charts import generate_bar_charts
from .likert_plots import generate_likert_plots
from .summary_stats import generate_summary_stats


def main():
    parser = argparse.ArgumentParser(description='Generate all plots for full sampling experiment')
    parser.add_argument('--only', type=str, choices=[
        'cdf_all', 'cdf_4subplot', 'cdf_by_topic', 'bar', 'likert', 'summary'
    ], help='Generate only specific plot type')
    args = parser.parse_args()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating plots for Full Sampling Experiment")
    print("=" * 60)
    
    if args.only:
        if args.only == 'cdf_all':
            plot_cdf_all_methods()
        elif args.only == 'cdf_4subplot':
            generate_cdf_4subplot_plots()
        elif args.only == 'cdf_by_topic':
            generate_cdf_by_topic_plots()
        elif args.only == 'bar':
            generate_bar_charts()
        elif args.only == 'likert':
            generate_likert_plots()
        elif args.only == 'summary':
            generate_summary_stats()
    else:
        # Generate all plots
        print("\n[1/6] CDF All Methods")
        plot_cdf_all_methods()
        
        print("\n[2/6] CDF 4-Subplot (per topic)")
        generate_cdf_4subplot_plots()
        
        print("\n[3/6] CDF By Topic (single plot)")
        generate_cdf_by_topic_plots()
        
        print("\n[4/6] Bar Charts")
        generate_bar_charts()
        
        print("\n[5/6] Likert Plots")
        generate_likert_plots()
        
        print("\n[6/6] Summary Statistics")
        generate_summary_stats()
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
