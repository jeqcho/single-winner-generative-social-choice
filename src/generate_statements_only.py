#!/usr/bin/env python3
"""
Generate statements only (no ranking or evaluation).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

from src.large_scale.persona_loader import load_persona_splits
from src.large_scale.generate_statements import generate_all_statements, save_statements

# Load environment
load_dotenv()

def load_topics(filepath: str):
    """Load topics from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def slugify(text: str) -> str:
    """Convert text to slug format."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]

def main():
    # Load personas (production: 900/50/50)
    print("Loading production personas (900/50/50)...")
    generative_personas, _, _ = load_persona_splits(test_mode=False)
    print(f"Loaded {len(generative_personas)} generative personas")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load topics
    topics = load_topics("data/topics.txt")
    print(f"Loaded {len(topics)} topics\n")
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate statements for each topic
    for i, topic in enumerate(topics, 1):
        topic_slug = slugify(topic)
        output_path = f"data/large_scale/prod/statements/{topic_slug}.json"
        
        # Check if already exists
        if os.path.exists(output_path):
            print(f"[{i}/{len(topics)}] ✓ {topic_slug[:40]}... (already exists)")
            continue
        
        print(f"[{i}/{len(topics)}] Generating statements for: {topic}")
        print(f"  Output: {output_path}")
        
        try:
            statements = generate_all_statements(topic, generative_personas, openai_client)
            save_statements(statements, topic_slug, output_dir="data/large_scale/prod/statements")
            print(f"  ✓ Generated {len(statements)} statements\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    print("=" * 80)
    print("✅ Statement generation complete!")
    print(f"Generated statements saved to: data/large_scale/prod/statements/")
    print("=" * 80)

if __name__ == "__main__":
    main()

