"""
Compute demographic statistics for persona clusters.

This script computes aggregate statistics for each cluster to enable
verification of GPT-generated cluster descriptions.
"""

import json
import re
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# GPT model for summaries
GPT_MODEL = "gpt-5.2"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_persona(text: str) -> Dict[str, Any]:
    """Extract fields from persona string using regex."""
    fields = {}
    
    # Extract age
    age_match = re.search(r'age:\s*(\d+)', text)
    if age_match:
        fields['age'] = int(age_match.group(1))
    
    # Extract sex
    sex_match = re.search(r'sex:\s*(\w+)', text)
    if sex_match:
        fields['sex'] = sex_match.group(1)
    
    # Extract race
    race_match = re.search(r'race:\s*([^\n]+)', text)
    if race_match:
        fields['race'] = race_match.group(1).strip()
    
    # Extract political views
    politics_match = re.search(r'political views:\s*([^\n]+)', text)
    if politics_match:
        fields['political_views'] = politics_match.group(1).strip()
    
    # Extract ideology
    ideology_match = re.search(r'ideology:\s*([^\n]+)', text)
    if ideology_match:
        fields['ideology'] = ideology_match.group(1).strip()
    
    # Extract religion
    religion_match = re.search(r'religion:\s*([^\n]+)', text)
    if religion_match:
        fields['religion'] = religion_match.group(1).strip()
    
    # Extract income - handle various formats
    income_match = re.search(r'income:\s*([^\n]+)', text)
    if income_match:
        fields['income'] = parse_income(income_match.group(1).strip())
    
    return fields


def parse_income(value: str) -> Optional[float]:
    """
    Parse income value, handling both numeric and range formats.
    Returns midpoint for ranges.
    """
    if not value or value.lower() in ['not applicable', 'n/a', 'none', '']:
        return None
    
    # Remove currency symbols and commas
    value = value.replace('$', '').replace(',', '').strip()
    
    # Check for range format (e.g., "20000-30000")
    range_match = re.match(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', value)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2
    
    # Try to parse as single number
    try:
        return float(value)
    except ValueError:
        return None


def get_age_bracket(age: int) -> str:
    """Convert age to 10-year bracket string."""
    decade = (age // 10) * 10
    return f"{decade}-{decade + 9}"


def compute_cluster_stats(
    personas: List[str],
    cluster_ids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Compute statistics for each cluster.
    
    Returns a dict mapping cluster_id to stats dict.
    """
    # Group personas by cluster
    clusters: Dict[int, List[Dict]] = {}
    for persona, cluster_id in zip(personas, cluster_ids):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        parsed = parse_persona(persona)
        clusters[cluster_id].append(parsed)
    
    # Compute stats for each cluster
    stats = {}
    for cluster_id in sorted(clusters.keys()):
        cluster_personas = clusters[cluster_id]
        stats[cluster_id] = compute_single_cluster_stats(cluster_personas)
    
    return stats


def compute_single_cluster_stats(personas: List[Dict]) -> Dict[str, Any]:
    """Compute statistics for a single cluster."""
    size = len(personas)
    
    # Age stats
    ages = [p['age'] for p in personas if 'age' in p]
    avg_age = np.mean(ages) if ages else None
    
    # Age bracket
    age_brackets = [get_age_bracket(a) for a in ages]
    bracket_counter = Counter(age_brackets)
    if bracket_counter:
        top_bracket, top_count = bracket_counter.most_common(1)[0]
        bracket_pct = top_count / len(ages) * 100
        age_bracket_str = f"{top_bracket} ({bracket_pct:.0f}%)"
    else:
        age_bracket_str = "N/A"
    
    # Sex stats
    sexes = [p['sex'] for p in personas if 'sex' in p]
    female_count = sum(1 for s in sexes if s and s.lower() == 'female')
    female_pct = female_count / len(sexes) * 100 if sexes else 0
    
    # Race stats
    races = [p['race'] for p in personas if 'race' in p]
    race_counter = Counter(races)
    if race_counter:
        top_race, top_count = race_counter.most_common(1)[0]
        # Shorten race name for display
        short_race = shorten_race(top_race)
        race_pct = top_count / len(races) * 100
        race_str = f"{short_race} ({race_pct:.0f}%)"
    else:
        race_str = "N/A"
    
    # Political views stats
    politics = [p['political_views'] for p in personas if 'political_views' in p]
    politics_counter = Counter(politics)
    if politics_counter:
        top_pol, top_count = politics_counter.most_common(1)[0]
        # Shorten political view for display
        short_pol = shorten_politics(top_pol)
        pol_pct = top_count / len(politics) * 100
        politics_str = f"{short_pol} ({pol_pct:.0f}%)"
    else:
        politics_str = "N/A"
    
    # Ideology stats
    ideologies = [p['ideology'] for p in personas if 'ideology' in p]
    ideology_counter = Counter(ideologies)
    if ideology_counter:
        top_ideo, top_count = ideology_counter.most_common(1)[0]
        # Shorten ideology for display
        short_ideo = shorten_ideology(top_ideo)
        ideo_pct = top_count / len(ideologies) * 100
        ideology_str = f"{short_ideo} ({ideo_pct:.0f}%)"
    else:
        ideology_str = "N/A"
    
    # Religion stats
    religions = [p['religion'] for p in personas if 'religion' in p]
    religion_counter = Counter(religions)
    if religion_counter:
        top_rel, top_count = religion_counter.most_common(1)[0]
        # Shorten religion for display
        short_rel = shorten_religion(top_rel)
        rel_pct = top_count / len(religions) * 100
        religion_str = f"{short_rel} ({rel_pct:.0f}%)"
    else:
        religion_str = "N/A"
    
    # Income stats (filter out negative/invalid values)
    incomes = [p['income'] for p in personas if p.get('income') is not None and p['income'] >= 0]
    if incomes:
        avg_income = np.mean(incomes)
        q1 = max(0, np.percentile(incomes, 25))  # Ensure Q1 is not negative
        q3 = np.percentile(incomes, 75)
        avg_income_str = format_income(avg_income)
        income_iqr_str = f"Q1: {format_income(q1)}, Q3: {format_income(q3)}"
    else:
        avg_income_str = "N/A"
        income_iqr_str = "N/A"
    
    return {
        'size': size,
        'avg_age': f"{avg_age:.0f}" if avg_age else "N/A",
        'age_bracket': age_bracket_str,
        'pct_female': f"{female_pct:.0f}%",
        'top_race': race_str,
        'politics': politics_str,
        'ideology': ideology_str,
        'religion': religion_str,
        'avg_income': avg_income_str,
        'income_iqr': income_iqr_str,
    }


def shorten_race(race: str) -> str:
    """Shorten race description for table display."""
    if 'White' in race:
        return 'White'
    if 'Black' in race or 'African American' in race:
        return 'Black'
    if 'Asian' in race:
        if 'Indian' in race:
            return 'Asian Indian'
        return 'Asian'
    if 'Two or More' in race:
        return 'Multiracial'
    if 'Some Other' in race:
        return 'Other'
    if 'Hispanic' in race or 'Latino' in race:
        return 'Hispanic'
    return race[:20] if len(race) > 20 else race


def shorten_politics(pol: str) -> str:
    """Shorten political view for table display."""
    pol_lower = pol.lower()
    if 'democrat' in pol_lower:
        return 'Democrat'
    if 'republican' in pol_lower:
        return 'Republican'
    if 'independent' in pol_lower:
        return 'Independent'
    if 'green' in pol_lower:
        return 'Green'
    if 'libertarian' in pol_lower:
        return 'Libertarian'
    if 'liberal' in pol_lower:
        return 'Liberal'
    if 'conservative' in pol_lower:
        return 'Conservative'
    if 'non-partisan' in pol_lower or 'nonpartisan' in pol_lower:
        return 'Non-partisan'
    if 'too young' in pol_lower or 'not applicable' in pol_lower:
        return 'N/A'
    return pol[:15] if len(pol) > 15 else pol


def shorten_ideology(ideo: str) -> str:
    """Shorten ideology for table display."""
    ideo_lower = ideo.lower()
    if 'liberal' in ideo_lower:
        return 'Liberal'
    if 'conservative' in ideo_lower:
        return 'Conservative'
    if 'progressive' in ideo_lower:
        return 'Progressive'
    if 'moderate' in ideo_lower:
        return 'Moderate'
    if 'libertarian' in ideo_lower:
        return 'Libertarian'
    return ideo[:15] if len(ideo) > 15 else ideo


def shorten_religion(rel: str) -> str:
    """Shorten religion for table display."""
    rel_lower = rel.lower()
    if 'unaffiliated' in rel_lower:
        return 'Unaffiliated'
    if 'protestant' in rel_lower:
        return 'Protestant'
    if 'catholic' in rel_lower:
        return 'Catholic'
    if 'jewish' in rel_lower:
        return 'Jewish'
    if 'muslim' in rel_lower or 'islam' in rel_lower:
        return 'Muslim'
    if 'buddhist' in rel_lower:
        return 'Buddhist'
    if 'hindu' in rel_lower:
        return 'Hindu'
    if 'other christian' in rel_lower:
        return 'Other Christian'
    if 'other non-christian' in rel_lower:
        return 'Other Non-Christian'
    return rel[:15] if len(rel) > 15 else rel


def format_income(value: float) -> str:
    """Format income value as $XXK or $X.XM."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    else:
        return f"${value/1_000:.0f}K"


def parse_cluster_descriptions(report_file: Path, k: int) -> Dict[int, str]:
    """
    Parse cluster descriptions from a report markdown file.
    
    Returns a dict mapping cluster_id to description text.
    """
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Find the section for this k value
    section_pattern = rf'## K={k} Clusters\s*\n\n\|.*?\n\|[-\s|]+\n(.*?)(?=\n## K=|\n!\[|$)'
    section_match = re.search(section_pattern, content, re.DOTALL)
    
    if not section_match:
        return {}
    
    table_content = section_match.group(1)
    
    # Parse each row
    descriptions = {}
    row_pattern = r'\|\s*(\d+)\s*\|\s*\d+\s*\|\s*(.*?)\s*\|[^|]+\|'
    for match in re.finditer(row_pattern, table_content):
        cluster_id = int(match.group(1))
        description = match.group(2).strip()
        descriptions[cluster_id] = description
    
    return descriptions


def generate_summary(description: str, client: OpenAI) -> str:
    """
    Generate a 1-sentence summary of a cluster description using GPT-5.2.
    """
    prompt = f"""Summarize the following cluster description in a single short phrase (under 80 characters), focusing on the key demographics (age, gender, occupation, political leaning). Do not include any quotes or punctuation at the end.

Description: {description}

One-phrase summary:"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=50,
        )
        summary = response.choices[0].message.content.strip()
        # Remove any trailing punctuation
        summary = summary.rstrip('.')
        return summary
    except Exception as e:
        logging.warning(f"Error generating summary: {e}")
        return "N/A"


def generate_summaries_for_k(
    descriptions: Dict[int, str],
    client: OpenAI
) -> Dict[int, str]:
    """Generate summaries for all clusters in a k value."""
    summaries = {}
    for cluster_id in sorted(descriptions.keys()):
        desc = descriptions[cluster_id]
        logging.info(f"  Generating summary for cluster {cluster_id}...")
        summaries[cluster_id] = generate_summary(desc, client)
    return summaries


def format_stats_table(
    stats: Dict[int, Dict],
    k: int,
    summaries: Optional[Dict[int, str]] = None
) -> str:
    """Generate markdown table string for cluster stats."""
    if summaries:
        lines = [
            f"### Enhanced Stats (K={k})",
            "",
            "| Cluster | Size | Avg Age | Age Bracket | % Female | Top Race | Politics | Ideology | Religion | Avg Income | Income IQR | Summary |",
            "|---------|------|---------|-------------|----------|----------|----------|----------|----------|------------|------------|---------|",
        ]
    else:
        lines = [
            f"### Enhanced Stats (K={k})",
            "",
            "| Cluster | Size | Avg Age | Age Bracket | % Female | Top Race | Politics | Ideology | Religion | Avg Income | Income IQR |",
            "|---------|------|---------|-------------|----------|----------|----------|----------|----------|------------|------------|",
        ]
    
    for cluster_id in sorted(stats.keys()):
        s = stats[cluster_id]
        row = f"| {cluster_id} | {s['size']} | {s['avg_age']} | {s['age_bracket']} | {s['pct_female']} | {s['top_race']} | {s['politics']} | {s['ideology']} | {s['religion']} | {s['avg_income']} | {s['income_iqr']} |"
        if summaries:
            summary = summaries.get(cluster_id, "N/A")
            row += f" {summary} |"
        lines.append(row)
    
    lines.append("")
    return "\n".join(lines)


def load_data(
    personas_file: Path,
    clusters_file: Path
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Load personas and cluster assignments."""
    with open(personas_file, 'r') as f:
        personas = json.load(f)
    
    with open(clusters_file, 'r') as f:
        clusters = json.load(f)
    
    return personas, clusters


def generate_all_tables(
    personas_file: Path,
    clusters_file: Path,
    k_values: List[int] = [5, 10, 20],
    report_file: Optional[Path] = None,
    generate_summaries: bool = False
) -> str:
    """Generate all stats tables for given k values."""
    personas, cluster_assignments = load_data(personas_file, clusters_file)
    
    # Initialize OpenAI client if generating summaries
    client = None
    if generate_summaries:
        client = OpenAI()
        logging.info("Initialized OpenAI client for summary generation")
    
    all_tables = []
    for k in k_values:
        k_str = str(k)
        if k_str in cluster_assignments:
            stats = compute_cluster_stats(personas, cluster_assignments[k_str])
            
            # Generate summaries if requested and report file is provided
            summaries = None
            if generate_summaries and report_file and report_file.exists():
                logging.info(f"Parsing descriptions for K={k}...")
                descriptions = parse_cluster_descriptions(report_file, k)
                if descriptions:
                    logging.info(f"Generating summaries for K={k}...")
                    summaries = generate_summaries_for_k(descriptions, client)
            
            table = format_stats_table(stats, k, summaries)
            all_tables.append(table)
    
    return "\n".join(all_tables)


def main():
    parser = argparse.ArgumentParser(
        description="Compute cluster statistics for persona clustering reports"
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        default=str(DATA_DIR / "personas" / "prod" / "full.json"),
        help="Path to personas JSON file"
    )
    parser.add_argument(
        "--clusters-file",
        type=str,
        default=str(DATA_DIR / "persona_embeddings" / "persona_clusters.json"),
        help="Path to cluster assignments JSON file"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
        help="Comma-separated list of k values to compute stats for"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Path to existing report file to extract descriptions from"
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Generate 1-sentence summaries using GPT-5.2"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (prints to stdout if not specified)"
    )
    
    args = parser.parse_args()
    
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    
    report_file = Path(args.report_file) if args.report_file else None
    
    tables = generate_all_tables(
        Path(args.personas_file),
        Path(args.clusters_file),
        k_values,
        report_file=report_file,
        generate_summaries=args.generate_summaries
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(tables)
        print(f"Stats written to {args.output}")
    else:
        print(tables)


if __name__ == "__main__":
    main()
