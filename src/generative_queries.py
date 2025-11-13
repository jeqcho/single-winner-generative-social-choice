"""
Generate statements from personas using OpenAI API.
"""

from typing import List, Dict
from openai import OpenAI


def generate_statements(topic: str, personas: List[Dict], openai_client: OpenAI = None) -> List[Dict]:
    """
    Generate statements from 10 personas on a given topic.
    
    Args:
        topic: The topic/question to generate statements about
        personas: List of 10 persona dictionaries
        openai_client: OpenAI client instance
    
    Returns:
        List of statements with persona attribution, each dict has:
        - persona: persona dict
        - statement: generated statement string
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    if len(personas) != 10:
        raise ValueError(f"Expected 10 personas, got {len(personas)}")
    
    statements = []
    
    for persona in personas:
        persona_desc = f"""Name: {persona.get('name', 'Unknown')}
Background: {persona.get('background', 'Not specified')}
Perspective: {persona.get('perspective', 'Not specified')}
Values: {persona.get('values', 'Not specified')}
Communication Style: {persona.get('communication_style', 'Not specified')}"""
        
        prompt = f"""You are {persona.get('name', 'a person')} with the following characteristics:

{persona_desc}

Given the topic: "{topic}"

Write a statement expressing your views on this topic. The statement should:
- Reflect your background, perspective, and values
- Be clear and substantive (2-4 sentences)
- Represent a genuine viewpoint someone with your characteristics might hold

Write only the statement, no additional commentary."""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        statement_text = response.choices[0].message.content.strip()
        
        statements.append({
            "persona": persona,
            "statement": statement_text
        })
    
    return statements

