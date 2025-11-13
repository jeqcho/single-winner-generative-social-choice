"""
Generate summaries from personas' perspectives using OpenAI API.
"""

from typing import List, Dict
from openai import OpenAI


def generate_summaries(statements: List[Dict], personas: List[Dict], openai_client: OpenAI = None) -> List[Dict]:
    """
    Generate summaries from each persona's perspective of all statements.
    
    Args:
        statements: List of statement dicts (from generative_queries)
        personas: List of 10 personas (same ones that generated statements)
        openai_client: OpenAI client instance
    
    Returns:
        List of summaries with persona attribution, each dict has:
        - persona: persona dict
        - summary: generated summary string
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    if len(personas) != 10:
        raise ValueError(f"Expected 10 personas, got {len(personas)}")
    
    if len(statements) != 10:
        raise ValueError(f"Expected 10 statements, got {len(statements)}")
    
    # Format all statements for context
    statements_text = "\n\n".join([
        f"Statement {i+1} (from {stmt['persona'].get('name', 'Unknown')}): {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    summaries = []
    
    for persona in personas:
        persona_desc = f"""Name: {persona.get('name', 'Unknown')}
Background: {persona.get('background', 'Not specified')}
Perspective: {persona.get('perspective', 'Not specified')}
Values: {persona.get('values', 'Not specified')}
Communication Style: {persona.get('communication_style', 'Not specified')}"""
        
        prompt = f"""You are {persona.get('name', 'a person')} with the following characteristics:

{persona_desc}

After a discussion, the following statements were made:

{statements_text}

From your perspective, write a summary that captures the essence of the discussion. This summary should:
- Reflect how you, with your background and perspective, would summarize the key points
- Be a reasonable alternative that could represent a consensus or bridging statement
- Be clear and substantive (3-5 sentences)
- Not simply repeat one statement, but synthesize the discussion from your viewpoint

Write only the summary, no additional commentary."""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        summary_text = response.choices[0].message.content.strip()
        
        summaries.append({
            "persona": persona,
            "summary": summary_text
        })
    
    return summaries

