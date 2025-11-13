"""
Generate diverse personas using OpenAI API.
"""

from typing import List, Dict
from openai import OpenAI


def generate_personas(n: int = 30, openai_client: OpenAI = None) -> List[Dict]:
    """
    Generate n diverse personas with different backgrounds and perspectives.
    
    Args:
        n: Number of personas to generate (default 30)
        openai_client: OpenAI client instance
    
    Returns:
        List of persona dictionaries with metadata (name, background, perspective, etc.)
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    prompt = f"""Generate {n} diverse personas with different backgrounds, perspectives, and viewpoints. 
Each persona should have:
- A name
- A brief background (occupation, education, life experience)
- Political/social perspective (liberal, conservative, moderate, progressive, etc.)
- Key values and priorities
- Communication style

Make them diverse in terms of:
- Age, gender, ethnicity
- Geographic location (urban, suburban, rural)
- Socioeconomic status
- Political orientation
- Professional background
- Life experiences

Return as a JSON object with a "personas" key containing an array where each persona is an object with fields: name, background, perspective, values, communication_style.

Format: {{"personas": [{{"name": "...", "background": "...", "perspective": "...", "values": "...", "communication_style": "..."}}, ...]}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse personas in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.9
        )
    except Exception:
        # Fallback if response_format not supported
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse personas in JSON format. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
    
    import json
    result = json.loads(response.choices[0].message.content)
    
    # Extract personas array from result
    personas = result.get("personas", [])
    if not isinstance(personas, list):
        # Fallback: try to find any list in the result
        for value in result.values():
            if isinstance(value, list):
                personas = value
                break
    
    # Ensure we have exactly n personas
    if len(personas) < n:
        # Generate additional personas if needed
        additional = n - len(personas)
        additional_prompt = f"""Generate {additional} more diverse personas with different backgrounds and perspectives.
Return as a JSON object with a "personas" key containing an array with the same format as before.
Format: {{"personas": [{{"name": "...", "background": "...", "perspective": "...", "values": "...", "communication_style": "..."}}, ...]}}"""
        
        try:
            additional_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse personas in JSON format. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": additional_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.9
            )
        except Exception:
            # Fallback if response_format not supported
            additional_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse personas in JSON format. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": additional_prompt}
                ],
                temperature=0.9
            )
        
        additional_result = json.loads(additional_response.choices[0].message.content)
        additional_personas = additional_result.get("personas", [])
        if not isinstance(additional_personas, list):
            for value in additional_result.values():
                if isinstance(value, list):
                    additional_personas = value
                    break
        
        personas.extend(additional_personas[:additional])
    
    return personas[:n]

