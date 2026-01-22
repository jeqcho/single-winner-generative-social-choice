"""
Verbalized sampling utilities for Alt3 and Alt4 statement generation.

Verbalized sampling is a prompting technique where the LLM generates multiple
diverse responses with associated probabilities. We use this to get more diverse
statements without needing multiple API calls.

The probabilities are just a prompting technique to encourage diversity -
we ignore them and use all 5 statements from each response.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_verbalized_response(response_text: str) -> List[str]:
    """
    Parse verbalized sampling response and return all statement texts.
    
    Expected format:
    <response>
    <text>Statement text here...</text>
    <probability>0.08</probability>
    </response>
    ... (5 total responses)
    
    Args:
        response_text: Raw response from LLM containing XML-style tags
        
    Returns:
        List of statement texts (typically 5, but may be fewer if parsing fails)
        
    Note:
        Probabilities are ignored - they're just a prompting technique for diversity.
        We use all statements returned.
    """
    # Pattern to match <response> blocks with <text> and <probability>
    pattern = r'<response>\s*<text>(.*?)</text>\s*<probability>[\d.]+</probability>\s*</response>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    # Clean up and return the statement texts
    statements = [text.strip() for text in matches]
    
    if len(statements) == 0:
        logger.warning(f"No statements parsed from response. Raw text: {response_text[:500]}...")
        # Try alternative patterns as fallback
        statements = _try_fallback_parsing(response_text)
    
    if len(statements) < 5:
        logger.warning(f"Expected 5 statements but got {len(statements)}")
    
    return statements


def _try_fallback_parsing(response_text: str) -> List[str]:
    """
    Try alternative parsing patterns if the main pattern fails.
    """
    statements = []
    
    # Try pattern with just <text> tags
    pattern_alt1 = r'<text>(.*?)</text>'
    matches = re.findall(pattern_alt1, response_text, re.DOTALL)
    if matches:
        statements = [text.strip() for text in matches]
        logger.info(f"Fallback pattern found {len(statements)} statements")
        return statements
    
    return statements


def get_verbalized_system_prompt() -> str:
    """
    Return the system prompt for verbalized sampling.
    """
    return """You are a helpful assistant that generates statements. Return only the statement text, no JSON or additional commentary. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10."""


def format_statements_for_context(statements: List[str], max_statements: int = 100) -> str:
    """
    Format a list of statements for inclusion in a prompt as context.
    """
    selected = statements[:max_statements]
    formatted_lines = []
    for i, stmt in enumerate(selected, 1):
        formatted_lines.append(f"{i}. {stmt}")
    return "\n".join(formatted_lines)
