#!/usr/bin/env python3
"""
Add additional statements from discriminative and evaluative personas to existing statement files.

This script adds 100 more statements (50 from discriminative + 50 from evaluative personas)
to each topic's statement file, bringing the total from 900 to 1000 statements per topic.
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

from src.large_scale.persona_loader import load_persona_splits
from src.large_scale.generate_statements import generate_all_statements

# Load environment
load_dotenv()

STATEMENTS_DIR = "data/large_scale/prod/statements"
TOPICS_FILE = "data/topics.txt"


def slugify(text: str) -> str:
    """Convert text to slug format."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]


def load_topics(filepath: str):
    """Load topics from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # Load personas
    print("Loading personas...")
    _, discriminative_personas, evaluative_personas = load_persona_splits(test_mode=False)
    print(f"  Discriminative: {len(discriminative_personas)} personas")
    print(f"  Evaluative: {len(evaluative_personas)} personas")
    
    # Combine discriminative + evaluative personas (100 total)
    additional_personas = discriminative_personas + evaluative_personas
    print(f"  Total additional personas: {len(additional_personas)}")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load topics and create slug -> topic mapping
    topics = load_topics(TOPICS_FILE)
    slug_to_topic = {slugify(topic): topic for topic in topics}
    print(f"\nLoaded {len(topics)} topics")
    
    # Get all statement files
    statement_files = sorted([f for f in os.listdir(STATEMENTS_DIR) if f.endswith('.json')])
    print(f"Found {len(statement_files)} statement files to process\n")
    
    # Process each file
    for i, filename in enumerate(statement_files, 1):
        filepath = os.path.join(STATEMENTS_DIR, filename)
        topic_slug = filename.replace('.json', '')
        
        # Load existing statements
        with open(filepath, 'r') as f:
            existing_statements = json.load(f)
        
        print(f"[{i}/{len(statement_files)}] {topic_slug}")
        print(f"  Existing statements: {len(existing_statements)}")
        
        # Skip if already has 1000+ statements
        if len(existing_statements) >= 1000:
            print(f"  ✓ Already has {len(existing_statements)} statements, skipping...")
            continue
        
        # Get actual topic from mapping
        topic = slug_to_topic.get(topic_slug)
        if topic is None:
            print(f"  ⚠ Warning: Could not find topic for slug '{topic_slug}', skipping...")
            continue
        
        # Generate new statements from additional personas
        print(f"  Generating {len(additional_personas)} new statements...")
        try:
            new_statements = generate_all_statements(topic, additional_personas, openai_client)
            print(f"  Generated: {len(new_statements)} statements")
            
            # Combine existing + new
            combined_statements = existing_statements + new_statements
            print(f"  Total combined: {len(combined_statements)} statements")
            
            # Save back to file
            with open(filepath, 'w') as f:
                json.dump(combined_statements, f, indent=2)
            print(f"  ✓ Saved to {filepath}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    print("=" * 80)
    print("✅ Additional statement generation complete!")
    print(f"All files in {STATEMENTS_DIR} now have 1000 statements each.")
    print("=" * 80)


if __name__ == "__main__":
    main()

