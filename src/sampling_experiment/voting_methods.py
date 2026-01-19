"""
Voting methods for the sampling experiment.

Includes:
- Traditional methods: Schulze, Borda, IRV, Plurality, Veto by Consumption
- ChatGPT methods: Select from P alternatives
- ChatGPT* methods: Select from all 100 alternatives
- ChatGPT** methods: Generate a new statement
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from votekit import RankProfile, RankBallot
from votekit.elections import Plurality, Borda, IRV, RankedPairs

from src.compute_pvc import compute_pvc
from .config import MODEL, TEMPERATURE, api_timer
from .single_call_ranking import insert_statement_into_ranking

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _preferences_to_votekit(preferences: List[List[str]]) -> Tuple[RankProfile, List[str]]:
    """
    Convert preference matrix to VoteKit format.
    
    Args:
        preferences: Matrix where preferences[rank][voter] is statement index
    
    Returns:
        Tuple of (RankProfile, candidate_names)
    """
    n_statements = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    candidates = [f"c{i}" for i in range(n_statements)]
    
    ballots = []
    for voter in range(n_voters):
        ranking = []
        for rank in range(n_statements):
            stmt_idx = int(preferences[rank][voter])
            ranking.append(frozenset([f"c{stmt_idx}"]))
        ballots.append(RankBallot(ranking=tuple(ranking)))
    
    profile = RankProfile(ballots=ballots, candidates=candidates)
    return profile, candidates


def _extract_winner(elected, prefix: str = "c") -> Optional[str]:
    """Extract winner index from VoteKit election result."""
    if elected and elected[0]:
        winner = list(elected[0])[0]
        if winner and isinstance(winner, str) and winner.startswith(prefix):
            return winner[1:]
    return None


# =============================================================================
# Traditional Voting Methods
# =============================================================================

def run_schulze(preferences: List[List[str]]) -> Dict:
    """Run Schulze/RankedPairs method (Condorcet)."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = RankedPairs(profile, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Schulze failed: {e}")
        return {"winner": None, "error": str(e)}


def run_borda(preferences: List[List[str]]) -> Dict:
    """Run Borda count."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = Borda(profile, m=1, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Borda failed: {e}")
        return {"winner": None, "error": str(e)}


def run_irv(preferences: List[List[str]]) -> Dict:
    """Run Instant Runoff Voting."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = IRV(profile, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"IRV failed: {e}")
        return {"winner": None, "error": str(e)}


def run_plurality(preferences: List[List[str]]) -> Dict:
    """Run Plurality voting."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = Plurality(profile, m=1, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Plurality failed: {e}")
        return {"winner": None, "error": str(e)}


def run_veto_by_consumption(preferences: List[List[str]]) -> Dict:
    """Run veto by consumption (PVC) method."""
    try:
        n_statements = len(preferences)
        alternatives = [str(i) for i in range(n_statements)]
        pvc_result = compute_pvc(preferences, alternatives)
        winner = pvc_result[0] if pvc_result else None
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Veto by consumption failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# ChatGPT Methods (Select from P alternatives)
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt(
    statements: List[Dict],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT baseline selection from P alternatives."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Which statement would be the best choice as a consensus/bridging statement?
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as JSON: {{"selected_statement_index": <index>}}
Where the value is the index (0-{n-1}) of the statement you select."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_rankings(
    statements: List[Dict],
    preferences: List[List[str]],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT with preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Format rankings (show first 10 voters)
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [preferences[rank][voter] for rank in range(len(preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}...")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement would be the best choice as a consensus/bridging statement?

Return your choice as JSON: {{"selected_statement_index": <index>}}
Where the value is the index (0-{n-1}) of the statement you select."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                    {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with rankings failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_personas(
    statements: List[Dict],
    personas: List[str],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT with persona descriptions."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(personas)
    
    # Format personas (truncate if too many)
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona[:500]}..."  # Truncate long personas
        for i, persona in enumerate(personas[:10])
    ])
    
    if n_voters > 10:
        personas_text += f"\n\n... and {n_voters - 10} more voters"
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are the {n_voters} voters who will be voting on these statements:

{personas_text}

Based on both the statements and the voter personas, which statement would be the best choice as a consensus/bridging statement?

Return your choice as JSON: {{"selected_statement_index": <index>}}
Where the value is the index (0-{n-1}) of the statement you select."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                    {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with personas failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# ChatGPT* Methods (Select from all 100 alternatives)
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_star(
    all_statements: List[Dict],
    sample_statements: List[Dict],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """
    Run ChatGPT* selection from all 100 alternatives.
    
    Shows sample statements for context but allows selection from all 100.
    """
    # Show sample statements as context
    sample_text = "\n\n".join([
        f"Sample Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    # Show all statements for selection
    all_text = "\n".join([
        f"{i}: {stmt['statement'][:200]}..."  # Truncate for token efficiency
        for i, stmt in enumerate(all_statements)
    ])
    
    n_all = len(all_statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are some sample statements from a discussion:

{sample_text}

Below are ALL {n_all} available statements you can choose from:

{all_text}

Which statement (from 0-{n_all-1}) would be the best choice as a consensus/bridging statement?
You may choose any statement, not just the samples shown above.

Return your choice as JSON: {{"selected_statement_index": <index>}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        # Validate winner is in range
        if not winner.isdigit() or int(winner) >= n_all:
            logger.warning(f"Invalid winner {winner}, returning None")
            return {"winner": None, "error": "Invalid selection"}
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT* failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_star_with_rankings(
    all_statements: List[Dict],
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT* with preference rankings from sample."""
    # Rankings from sample
    n_voters = len(sample_preferences[0]) if sample_preferences else 0
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [sample_preferences[rank][voter] for rank in range(len(sample_preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}...")
    
    rankings_text = "\n".join(rankings_summary)
    
    # All statements
    all_text = "\n".join([
        f"{i}: {stmt['statement'][:200]}..."
        for i, stmt in enumerate(all_statements)
    ])
    
    n_all = len(all_statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are preference rankings from {n_voters} voters (rankings are over a sample of statements):

{rankings_text}

Below are ALL {n_all} available statements you can choose from:

{all_text}

Which statement (from 0-{n_all-1}) would be the best choice as a consensus/bridging statement?

Return your choice as JSON: {{"selected_statement_index": <index>}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        if not winner.isdigit() or int(winner) >= n_all:
            return {"winner": None, "error": "Invalid selection"}
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT* with rankings failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_star_with_personas(
    all_statements: List[Dict],
    sample_personas: List[str],
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT* with persona descriptions from sample."""
    # Personas from sample
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona[:500]}..."
        for i, persona in enumerate(sample_personas[:10])
    ])
    
    if len(sample_personas) > 10:
        personas_text += f"\n\n... and {len(sample_personas) - 10} more voters"
    
    # All statements
    all_text = "\n".join([
        f"{i}: {stmt['statement'][:200]}..."
        for i, stmt in enumerate(all_statements)
    ])
    
    n_all = len(all_statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Here are {len(sample_personas)} voters who will be voting:

{personas_text}

Below are ALL {n_all} available statements you can choose from:

{all_text}

Which statement (from 0-{n_all-1}) would be the best choice as a consensus/bridging statement for these voters?

Return your choice as JSON: {{"selected_statement_index": <index>}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        if not winner.isdigit() or int(winner) >= n_all:
            return {"winner": None, "error": "Invalid selection"}
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT* with personas failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# ChatGPT** Methods (Generate new statement)
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_new_statement(
    sample_statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Optional[str]:
    """Generate a new consensus statement."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    system_prompt = "You are a helpful assistant that generates consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Topic: {topic}

Here are some existing statements from a discussion:

{statements_text}

Generate a NEW statement that could serve as a better consensus/bridging statement.
The statement should:
- Represent a reasonable middle ground that could satisfy diverse perspectives
- Be different from the existing statements but address the same topic
- Be clear and substantive (2-4 sentences)

Return your new statement as JSON: {{"new_statement": "<your statement>"}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("new_statement")
    except Exception as e:
        logger.error(f"Generate new statement failed: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_new_statement_with_rankings(
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Optional[str]:
    """Generate a new consensus statement using preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    n_voters = len(sample_preferences[0]) if sample_preferences else 0
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [sample_preferences[rank][voter] for rank in range(len(sample_preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}...")
    
    rankings_text = "\n".join(rankings_summary)
    
    system_prompt = "You are a helpful assistant that generates consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Topic: {topic}

Here are existing statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on these rankings, generate a NEW statement that could serve as a better consensus/bridging statement.
The statement should:
- Consider what aspects are most preferred by voters
- Craft a statement that might satisfy more people
- Be clear and substantive (2-4 sentences)

Return your new statement as JSON: {{"new_statement": "<your statement>"}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("new_statement")
    except Exception as e:
        logger.error(f"Generate new statement with rankings failed: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_new_statement_with_personas(
    sample_statements: List[Dict],
    sample_personas: List[str],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Optional[str]:
    """Generate a new consensus statement using persona descriptions."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona[:500]}..."
        for i, persona in enumerate(sample_personas[:10])
    ])
    
    if len(sample_personas) > 10:
        personas_text += f"\n\n... and {len(sample_personas) - 10} more voters"
    
    system_prompt = "You are a helpful assistant that generates consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Topic: {topic}

Here are existing statements from a discussion:

{statements_text}

Here are the voters who will be voting:

{personas_text}

Based on these voter personas, generate a NEW statement that could serve as a better consensus/bridging statement.
The statement should:
- Consider the diverse perspectives represented
- Craft a statement that might satisfy more people
- Be clear and substantive (2-4 sentences)

Return your new statement as JSON: {{"new_statement": "<your statement>"}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("new_statement")
    except Exception as e:
        logger.error(f"Generate new statement with personas failed: {e}")
        return None


def run_chatgpt_double_star(
    sample_statements: List[Dict],
    all_statements: List[Dict],
    sample_personas: List[str],
    full_preferences: List[List[str]],
    voter_indices: List[int],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """
    Run ChatGPT** which generates a new statement.
    
    Returns the new statement text and the updated preference profile.
    """
    new_statement = generate_new_statement(sample_statements, topic, openai_client, model, temperature)
    
    if new_statement is None:
        return {"winner": None, "new_statement": None, "error": "Failed to generate statement"}
    
    return {
        "winner": str(len(all_statements)),  # New statement index
        "new_statement": new_statement,
        "is_new": True
    }


def run_chatgpt_double_star_with_rankings(
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    all_statements: List[Dict],
    sample_personas: List[str],
    full_preferences: List[List[str]],
    voter_indices: List[int],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT** with rankings to generate a new statement."""
    new_statement = generate_new_statement_with_rankings(
        sample_statements, sample_preferences, topic, openai_client, model, temperature
    )
    
    if new_statement is None:
        return {"winner": None, "new_statement": None, "error": "Failed to generate statement"}
    
    return {
        "winner": str(len(all_statements)),
        "new_statement": new_statement,
        "is_new": True
    }


def run_chatgpt_double_star_with_personas(
    sample_statements: List[Dict],
    sample_personas: List[str],
    all_statements: List[Dict],
    full_preferences: List[List[str]],
    voter_indices: List[int],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """Run ChatGPT** with personas to generate a new statement."""
    new_statement = generate_new_statement_with_personas(
        sample_statements, sample_personas, topic, openai_client, model, temperature
    )
    
    if new_statement is None:
        return {"winner": None, "new_statement": None, "error": "Failed to generate statement"}
    
    return {
        "winner": str(len(all_statements)),
        "new_statement": new_statement,
        "is_new": True
    }


def insert_new_statement_into_rankings(
    new_statement: str,
    all_statements: List[Dict],
    voter_personas: List[str],
    voter_indices: List[int],
    full_preferences: List[List[str]],
    topic: str,
    openai_client: OpenAI,
    max_workers: int = 100,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> List[List[str]]:
    """
    Insert a new statement into the rankings of sampled voters.
    
    Re-queries each voter to find where to insert the new statement.
    
    Args:
        new_statement: The newly generated statement text
        all_statements: Original list of statement dicts
        voter_personas: Persona strings for the K sampled voters
        voter_indices: Indices of sampled voters in the full preference matrix
        full_preferences: Full 100x100 preference matrix
        topic: Topic string
        openai_client: OpenAI client
        max_workers: Max parallel workers
        model: Model to use
        temperature: Temperature
    
    Returns:
        Updated preference matrix with new statement (101 alternatives)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    n_voters = len(voter_indices)
    n_alts = len(all_statements)
    new_idx = n_alts  # Index for new statement
    
    logger.info(f"Inserting new statement into rankings for {n_voters} voters...")
    
    def process_voter(args):
        """Process a single voter to insert new statement."""
        local_idx, voter_idx, persona = args
        
        # Get this voter's current ranking
        current_ranking = [int(full_preferences[rank][voter_idx]) 
                         for rank in range(n_alts)]
        
        # Insert new statement
        new_ranking = insert_statement_into_ranking(
            persona, current_ranking, all_statements, new_statement,
            topic, openai_client, model, temperature
        )
        
        return local_idx, new_ranking
    
    # Process all voters
    updated_rankings = [None] * n_voters
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, (i, voter_idx, voter_personas[i])): i
            for i, voter_idx in enumerate(voter_indices)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Inserting statement", unit="voter"):
            local_idx, new_ranking = future.result()
            updated_rankings[local_idx] = new_ranking
    
    # Convert to preference matrix format [rank][voter]
    n_total_alts = n_alts + 1
    updated_preferences = []
    for rank in range(n_total_alts):
        rank_row = []
        for voter in range(n_voters):
            alt_idx = updated_rankings[voter][rank]
            rank_row.append(str(alt_idx))
        updated_preferences.append(rank_row)
    
    return updated_preferences


# =============================================================================
# ChatGPT*** Methods (Generate blind bridging statement - no context)
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_bridging_statement_no_context(
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Optional[str]:
    """
    Generate a bridging statement given ONLY the topic (no existing statements).
    
    This is the ChatGPT*** method - completely blind generation.
    
    Args:
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Generated bridging statement (2-4 sentences), or None if failed
    """
    system_prompt = "You are a helpful assistant that generates bridging statements. Return ONLY valid JSON."
    
    user_prompt = f"""Given the topic: "{topic}"

Generate a bridging statement that could serve as a consensus position on this topic.
The statement should:
- Represent a reasonable middle ground that could satisfy diverse perspectives
- Acknowledge different viewpoints while finding common ground
- Be clear and substantive (2-4 sentences)

Return your statement as JSON: {{"bridging_statement": "<your statement>"}}"""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": "minimal"},
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("bridging_statement")
    except Exception as e:
        logger.error(f"Generate bridging statement (no context) failed: {e}")
        return None


def run_chatgpt_triple_star(
    topic: str,
    all_statements: List[Dict],
    voter_personas: List[str],
    full_preferences: List[List[str]],
    openai_client: OpenAI,
    n_generations: int = 5,
    max_workers: int = 100,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> Dict:
    """
    Run ChatGPT*** which generates blind bridging statements (no context).
    
    Generates n_generations statements, inserts each into all voters' rankings,
    computes epsilon for each with m=100, and returns the average epsilon.
    
    Args:
        topic: The topic/question being discussed
        all_statements: All 100 statement dicts
        voter_personas: All 100 voter persona strings
        full_preferences: Full 100x100 preference matrix
        openai_client: OpenAI client instance
        n_generations: Number of statements to generate (default: 5)
        max_workers: Max parallel workers for insertion
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Dict with average epsilon, individual epsilons, and generated statements
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from src.sampling_experiment.epsilon_calculator import compute_epsilon_for_new_statement
    
    n_voters = len(voter_personas)
    n_alts = len(all_statements)
    
    logger.info(f"ChatGPT***: Generating {n_generations} blind bridging statements...")
    
    generated_statements = []
    epsilons = []
    
    for gen_idx in range(n_generations):
        logger.info(f"  Generation {gen_idx + 1}/{n_generations}...")
        
        # Generate blind statement
        new_statement = generate_bridging_statement_no_context(
            topic, openai_client, model, temperature
        )
        
        if new_statement is None:
            logger.warning(f"  Generation {gen_idx + 1} failed, skipping...")
            continue
        
        generated_statements.append(new_statement)
        logger.info(f"  Generated: {new_statement[:100]}...")
        
        # Insert into all 100 voters' rankings
        logger.info(f"  Inserting into {n_voters} voters' rankings...")
        
        def process_voter(args):
            """Process a single voter to insert new statement."""
            voter_idx, persona = args
            
            # Get this voter's current ranking
            current_ranking = [int(full_preferences[rank][voter_idx]) 
                             for rank in range(n_alts)]
            
            # Insert new statement
            new_ranking = insert_statement_into_ranking(
                persona, current_ranking, all_statements, new_statement,
                topic, openai_client, model, temperature
            )
            
            return voter_idx, new_ranking
        
        updated_rankings = [None] * n_voters
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_voter, (i, voter_personas[i])): i
                for i in range(n_voters)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"  Inserting gen {gen_idx + 1}", unit="voter"):
                voter_idx, new_ranking = future.result()
                updated_rankings[voter_idx] = new_ranking
        
        # Convert to preference matrix format [rank][voter]
        n_total_alts = n_alts + 1
        updated_preferences = []
        for rank in range(n_total_alts):
            rank_row = []
            for voter in range(n_voters):
                alt_idx = updated_rankings[voter][rank]
                rank_row.append(str(alt_idx))
            updated_preferences.append(rank_row)
        
        # Compute epsilon with m=100 (no veto power for new statement)
        epsilon = compute_epsilon_for_new_statement(updated_preferences, n_alts)
        epsilons.append(epsilon)
        logger.info(f"  Epsilon for generation {gen_idx + 1}: {epsilon:.4f}")
    
    if not epsilons:
        return {
            "winner": None,
            "epsilon": None,
            "epsilons": [],
            "statements": [],
            "error": "All generations failed"
        }
    
    avg_epsilon = sum(e for e in epsilons if e is not None) / len([e for e in epsilons if e is not None])
    
    logger.info(f"ChatGPT***: Average epsilon across {len(epsilons)} generations: {avg_epsilon:.4f}")
    
    return {
        "winner": "generated",  # Not a specific index
        "epsilon": avg_epsilon,
        "epsilons": epsilons,
        "statements": generated_statements,
        "is_new": True,
        "n_generations": len(generated_statements)
    }
