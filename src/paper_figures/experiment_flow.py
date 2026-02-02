"""
Generate publication-quality experiment flow diagram for LaTeX paper.

Creates a clean flowchart showing the three phases of the experiment:
1. Data Generation (personas, topics, alternative distributions)
2. Preference Building (iterative ranking → preference matrix)
3. Winner Selection (traditional + GPT methods)
4. Evaluation (critical epsilon from PVC)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def create_experiment_flow_diagram(output_dir: Path = None):
    """
    Create a publication-quality experiment flow diagram.
    
    Args:
        output_dir: Directory to save outputs (default: outputs/paper/)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up figure with clean styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.linewidth': 0.5,
        'text.usetex': False,  # Set to True if LaTeX is available
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors for phases (muted, professional palette)
    colors = {
        'input': '#E8E8E8',      # Light gray
        'phase1': '#D4E6F1',     # Light blue
        'phase2': '#D5F5E3',     # Light green
        'phase3': '#FCF3CF',     # Light yellow
        'eval': '#FADBD8',       # Light red/pink
        'border': '#2C3E50',     # Dark blue-gray
    }
    
    # Helper function to draw boxes
    def draw_box(x, y, width, height, text, color, fontsize=9, bold=False, 
                 text_color='black', align='center'):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor=colors['border'],
            linewidth=1.2
        )
        ax.add_patch(box)
        
        weight = 'bold' if bold else 'normal'
        va = 'center'
        ha = 'center' if align == 'center' else 'left'
        text_x = x + width/2 if align == 'center' else x + 0.1
        
        ax.text(text_x, y + height/2, text, 
                fontsize=fontsize, fontweight=weight, color=text_color,
                ha=ha, va=va, wrap=True)
    
    # Helper function to draw arrows
    def draw_arrow(start, end, color=colors['border'], style='->'):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style,
            color=color,
            linewidth=1.5,
            mutation_scale=12,
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow)
    
    # ========== INPUT DATA (Top) ==========
    # Personas box
    draw_box(0.5, 8.8, 2, 0.8, "815 Adult\nPersonas", colors['input'], 
             fontsize=9, bold=True)
    
    # Topics box
    draw_box(3.5, 8.8, 2, 0.8, "13 Policy\nTopics", colors['input'], 
             fontsize=9, bold=True)
    
    # ========== PHASE 1: Data Generation ==========
    # Phase label
    ax.text(0.3, 7.9, "Phase 1: Data Generation", fontsize=11, fontweight='bold',
            color=colors['border'])
    
    # Voter Sampling
    draw_box(0.5, 6.8, 2.2, 0.9, 
             "Voter Sampling\n• Uniform (n=100)\n• Clustered by ideology", 
             colors['phase1'], fontsize=8)
    
    # Statement Generation
    draw_box(3.3, 6.8, 3.2, 0.9,
             "Statement Generation (gpt-5-mini)\n• Alt1: Persona only\n• Alt2: Persona + context\n• Alt3: Context only\n• Alt4: Blind",
             colors['phase1'], fontsize=8)
    
    # Arrows from inputs to Phase 1
    draw_arrow((1.5, 8.8), (1.5, 7.7))
    draw_arrow((4.5, 8.8), (4.5, 7.7))
    
    # ========== PHASE 2: Preference Building ==========
    # Phase label
    ax.text(0.3, 5.9, "Phase 2: Preference Building", fontsize=11, fontweight='bold',
            color=colors['border'])
    
    # Iterative Ranking box
    draw_box(0.5, 4.6, 3.5, 1.1,
             "Iterative Ranking (gpt-5-mini)\n• 5 rounds of top-K/bottom-K\n• 4-letter hash identifiers\n• Per-round shuffling",
             colors['phase2'], fontsize=8)
    
    # Preference Matrix box
    draw_box(5, 4.6, 2.5, 1.1,
             "100×100\nPreference\nMatrix",
             colors['phase2'], fontsize=9, bold=True)
    
    # Arrows
    draw_arrow((1.6, 6.8), (1.6, 5.7))
    draw_arrow((4.9, 6.8), (2.2, 5.7))
    draw_arrow((4, 5.15), (5, 5.15))
    
    # ========== PHASE 3: Winner Selection ==========
    # Phase label
    ax.text(0.3, 3.8, "Phase 3: Winner Selection", fontsize=11, fontweight='bold',
            color=colors['border'])
    
    # Traditional Methods
    draw_box(0.5, 2.4, 2.3, 1.2,
             "Traditional Methods\n• Schulze\n• Borda\n• IRV\n• Plurality\n• VBC",
             colors['phase3'], fontsize=8)
    
    # GPT Selection Methods
    draw_box(3.1, 2.4, 2.4, 1.2,
             "GPT Selection (gpt-5-mini)\n• GPT: select from P\n• GPT*: select from 100\n(+Rank, +Pers variants)",
             colors['phase3'], fontsize=8)
    
    # GPT Generation Methods
    draw_box(5.8, 2.4, 2.6, 1.2,
             "GPT Generation (gpt-5-mini)\n• GPT**: generate new\n  (given P statements)\n• GPT***: blind generation",
             colors['phase3'], fontsize=8)
    
    # Arrows from Phase 2 to Phase 3
    draw_arrow((2.25, 4.6), (1.6, 3.6))
    draw_arrow((6.25, 4.6), (4.3, 3.6))
    draw_arrow((6.25, 4.6), (7.1, 3.6))
    
    # ========== EVALUATION ==========
    # Phase label
    ax.text(0.3, 1.6, "Evaluation", fontsize=11, fontweight='bold',
            color=colors['border'])
    
    # Epsilon computation box
    draw_box(2, 0.4, 5, 1.0,
             "Critical Epsilon (ε*) from Proportional Veto Core\nLower ε* = better consensus (statement more broadly acceptable)",
             colors['eval'], fontsize=9, bold=False)
    
    # Arrows to evaluation
    draw_arrow((1.6, 2.4), (3.5, 1.4))
    draw_arrow((4.3, 2.4), (4.5, 1.4))
    draw_arrow((7.1, 2.4), (5.5, 1.4))
    
    # ========== FACTORIAL DESIGN ANNOTATION ==========
    # Box on right side showing the factorial design
    draw_box(7.2, 6.5, 2.5, 3.0,
             "Factorial Design\n\n4 Alternative Dists\n× 2 Voter Dists\n× 13 Topics\n× 10 Replications\n× 5 Mini-reps\n\n= 5,200 conditions",
             colors['input'], fontsize=8, bold=False)
    
    # Title
    ax.text(5, 9.7, "Experiment Overview", fontsize=14, fontweight='bold',
            ha='center', color=colors['border'])
    
    # Save outputs
    pdf_path = output_dir / "experiment_flow.pdf"
    png_path = output_dir / "experiment_flow.png"
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    
    return pdf_path, png_path


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate experiment flow diagram for paper"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory (default: outputs/paper/)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    create_experiment_flow_diagram(output_dir)


if __name__ == "__main__":
    main()
