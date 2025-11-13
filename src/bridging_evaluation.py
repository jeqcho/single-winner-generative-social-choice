"""
Evaluate bridging quality of PVC result using personas.
"""

from typing import List, Dict
from openai import OpenAI


def evaluate_bridging(
    pvc_result: List[str],
    summaries: List[Dict],
    statements: List[Dict],
    personas: List[Dict],
    openai_client: OpenAI = None
) -> List[Dict]:
    """
    Evaluate whether PVC statement is a good bridging/consensus statement.
    
    Args:
        pvc_result: List of summary indices (as strings) in the PVC
        summaries: List of summary dicts
        statements: List of original statement dicts
        personas: List of 10 personas (from bridging evaluation group)
        openai_client: OpenAI client instance
    
    Returns:
        List of evaluation results per persona
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    if len(personas) != 10:
        raise ValueError(f"Expected 10 personas, got {len(personas)}")
    
    # Get the PVC summary (use first one if multiple)
    pvc_idx = int(pvc_result[0]) if pvc_result else None
    if pvc_idx is None or pvc_idx >= len(summaries):
        raise ValueError(f"Invalid PVC index: {pvc_idx}")
    
    pvc_summary = summaries[pvc_idx]
    
    # Format context
    statements_text = "\n\n".join([
        f"Statement {i+1} (from {stmt['persona'].get('name', 'Unknown')}): {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    summaries_text = "\n\n".join([
        f"Summary {i}: {summary['summary']}"
        for i, summary in enumerate(summaries)
    ])
    
    evaluations = []
    
    for persona in personas:
        persona_desc = f"""Name: {persona.get('name', 'Unknown')}
Background: {persona.get('background', 'Not specified')}
Perspective: {persona.get('perspective', 'Not specified')}
Values: {persona.get('values', 'Not specified')}
Communication Style: {persona.get('communication_style', 'Not specified')}"""
        
        prompt = f"""You are {persona.get('name', 'a person')} with the following characteristics:

{persona_desc}

Original statements from the discussion:
{statements_text}

All summaries generated:
{summaries_text}

The Proportional Veto Core (PVC) selected this summary as the consensus/bridging statement:
Summary {pvc_idx}: {pvc_summary['summary']}

Evaluate whether this PVC-selected summary is a good bridging/consensus statement. Consider:
1. Does it effectively bridge different perspectives?
2. Does it represent a reasonable consensus?
3. How well does it capture the essence of the discussion?
4. Would you consider this a good choice for a final statement?

Return your evaluation as a JSON object with this format:
{{
    "is_good_bridging": true/false,
    "rating": 1-10,
    "reasoning": "your explanation"
}}

Return only the JSON, no additional text."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.6
            )
        except Exception:
            # Fallback if response_format not supported
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        evaluations.append({
            "persona": persona,
            "evaluation": result
        })
    
    return evaluations

