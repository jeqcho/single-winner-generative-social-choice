"""
Recalculate ChatGPT** epsilons after fixing the total_vertices formula bug.

The bug was:
    total_vertices = 2 * m_for_veto * n - n  (wrong)
    
Should be:
    total_vertices = m_for_veto * n + (m_actual - 1) * n  (correct)

With m_for_veto=100 and m_actual=101:
    Wrong:   2 * 100 * n - n = 199n
    Correct: 100 * n + 100 * n = 200n
    
The epsilon error is exactly: n / (m_for_veto * n) = 1/100 = 0.01

So: epsilon_correct = epsilon_wrong + 0.01
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Epsilon correction factor
EPSILON_CORRECTION = 0.01

# Methods that need correction (ChatGPT** variants)
CHATGPT_DOUBLE_STAR_METHODS = [
    'chatgpt_double_star',
    'chatgpt_double_star_rankings',
    'chatgpt_double_star_personas'
]


def recalculate_epsilons_for_topic(topic_dir: Path) -> dict:
    """
    Recalculate epsilons for all samples in a topic directory.
    
    Returns:
        Dict with statistics about the corrections made
    """
    stats = {
        'files_updated': 0,
        'epsilons_corrected': 0,
        'corrections': []
    }
    
    for rep_dir in sorted(topic_dir.glob('rep*')):
        for sample_dir in sorted(rep_dir.glob('k*_p*')):
            results_file = sample_dir / 'results.json'
            
            if not results_file.exists():
                continue
            
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            modified = False
            
            # Correct ChatGPT** epsilons
            for method in CHATGPT_DOUBLE_STAR_METHODS:
                if method in results and results[method].get('epsilon') is not None:
                    old_eps = results[method]['epsilon']
                    new_eps = old_eps + EPSILON_CORRECTION
                    results[method]['epsilon'] = new_eps
                    modified = True
                    stats['epsilons_corrected'] += 1
                    stats['corrections'].append({
                        'file': str(results_file.relative_to(topic_dir)),
                        'method': method,
                        'old': old_eps,
                        'new': new_eps
                    })
            
            if modified:
                # Save updated results
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                stats['files_updated'] += 1
                logger.info(f"Updated {results_file.relative_to(topic_dir)}")
    
    return stats


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent.parent / 'outputs' / 'sampling_experiment' / 'data'
    
    if not base_dir.exists():
        logger.error(f"Data directory not found: {base_dir}")
        return
    
    # Process all topics
    for topic_dir in sorted(base_dir.iterdir()):
        if topic_dir.is_dir():
            logger.info(f"\nProcessing topic: {topic_dir.name}")
            stats = recalculate_epsilons_for_topic(topic_dir)
            
            logger.info(f"  Files updated: {stats['files_updated']}")
            logger.info(f"  Epsilons corrected: {stats['epsilons_corrected']}")
            
            # Show sample corrections
            if stats['corrections']:
                logger.info("  Sample corrections:")
                for corr in stats['corrections'][:3]:
                    logger.info(f"    {corr['method']}: {corr['old']:.4f} -> {corr['new']:.4f}")
                if len(stats['corrections']) > 3:
                    logger.info(f"    ... and {len(stats['corrections']) - 3} more")
    
    logger.info("\nRecalculation complete!")


if __name__ == '__main__':
    main()
