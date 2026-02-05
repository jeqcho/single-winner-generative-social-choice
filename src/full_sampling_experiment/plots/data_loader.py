"""
Data loading utilities for plotting.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from .config import DATA_DIR


def load_results_for_topics(topic_dirs: List[Path]) -> Dict[str, List[float]]:
    """Load results from specified topic directories."""
    all_results = defaultdict(list)
    
    for topic_dir in topic_dirs:
        for rep_dir in sorted(topic_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith('rep'):
                continue
            for sample_dir in sorted(rep_dir.iterdir()):
                if not sample_dir.is_dir() or not sample_dir.name.startswith('sample'):
                    continue
                results_file = sample_dir / 'results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)
                    for method, data in results.items():
                        if 'epsilon' in data:
                            all_results[method].append(data['epsilon'])
    
    return dict(all_results)


def load_all_results() -> Dict[str, List[float]]:
    """Load results from all topics."""
    topic_dirs = [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
    return load_results_for_topics(topic_dirs)


def load_results_by_topic() -> Dict[str, Dict[str, List[float]]]:
    """Load results grouped by topic."""
    topic_results = {}
    
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        topic_results[topic_dir.name] = load_results_for_topics([topic_dir])
    
    return topic_results


def load_random_epsilons() -> List[float]:
    """Load random baseline epsilons from precomputed data."""
    random_epsilons = []
    
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        for rep_dir in sorted(topic_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith('rep'):
                continue
            precomputed_file = rep_dir / 'precomputed_epsilons.json'
            if precomputed_file.exists():
                with open(precomputed_file) as f:
                    precomputed = json.load(f)
                for v in precomputed.values():
                    if v is not None:
                        random_epsilons.append(float(v))
    
    return random_epsilons


def load_likert_scores() -> Dict[str, List[int]]:
    """Load Likert scores from all topics (rep0 only)."""
    topic_scores = {}
    
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        likert_file = topic_dir / "rep0" / "likert_scores.json"
        if likert_file.exists():
            with open(likert_file) as f:
                data = json.load(f)
            import numpy as np
            topic_scores[topic_dir.name] = np.array(data["scores"]).flatten().tolist()
    
    return topic_scores


def get_topic_dirs() -> List[Path]:
    """Get list of all topic directories."""
    return [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
