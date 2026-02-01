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
from .config import GENERATIVE_VOTING_MODEL, GENERATIVE_VOTING_REASONING, RANKING_MODEL, TEMPERATURE, api_timer, build_api_metadata

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


def filter_persona(persona: str) -> str:
    """
    Extract key demographic fields from a persona string.
    
    Filters persona to only include: age, sex, race, education,
    occupation category, political views, religion.
    
    Args:
        persona: Full persona string with newline-separated key: value pairs
        
    Returns:
        Filtered persona string with only the 7 key fields
    """
    fields_to_keep = {
        'age', 'sex', 'race', 'education', 
        'occupation category', 'political views', 'religion'
    }
    lines = []
    for line in persona.split('\n'):
        if ':' in line:
            key = line.split(':')[0].strip().lower()
            if key in fields_to_keep:
                # Rename 'occupation category' to 'occupation'
                if key == 'occupation category':
                    line = 'occupation:' + line.split(':', 1)[1]
                lines.append(line)
    return '\n'.join(lines)


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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_select",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Dict:
    """Run ChatGPT with preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Format rankings (show all voters with full rankings)
    rankings_summary = []
    for voter in range(n_voters):
        ranking = [preferences[rank][voter] for rank in range(len(preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking)}")
    
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_select_rankings",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_rankings",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Dict:
    """Run ChatGPT with persona descriptions."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(personas)
    
    # Format personas (filter to key demographic fields for all voters)
    personas_text = "\n\n".join([
        f"Voter {i+1}: {filter_persona(persona)}"
        for i, persona in enumerate(personas)
    ])
    
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_select_personas",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_personas",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Dict:
    """
    Run ChatGPT* selection from all 100 alternatives.
    
    Shows all statements with topic context for selection.
    """
    # Show all statements for selection (full text)
    all_text = "\n".join([
        f"{i}: {stmt['statement']}"
        for i, stmt in enumerate(all_statements)
    ])
    
    n_all = len(all_statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON."
    
    user_prompt = f"""Topic: {topic}

A group of participants submitted the following {n_all} statements on this topic:

{all_text}

Select the statement that would best serve as a consensus or bridging position - one that:
- Engages substantively with the topic
- Could be acceptable to participants with diverse viewpoints
- Avoids extreme or polarizing framing

Return your choice as JSON: {{"selected_statement_index": <index>}}
Where the value is the index (0-{n_all-1}) of the statement you select."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_star_select",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_star",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Dict:
    """Run ChatGPT* with preference rankings from sample."""
    # Rankings from sample (show all voters with full rankings)
    n_voters = len(sample_preferences[0]) if sample_preferences else 0
    rankings_summary = []
    for voter in range(n_voters):
        ranking = [sample_preferences[rank][voter] for rank in range(len(sample_preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking)}")
    
    rankings_text = "\n".join(rankings_summary)
    
    # All statements (full text)
    all_text = "\n".join([
        f"{i}: {stmt['statement']}"
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_star_select_rankings",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_star_rankings",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Dict:
    """Run ChatGPT* with persona descriptions from sample."""
    # Personas from sample (filter to key demographic fields for all voters)
    personas_text = "\n\n".join([
        f"Voter {i+1}: {filter_persona(persona)}"
        for i, persona in enumerate(sample_personas)
    ])
    
    # All statements (full text)
    all_text = "\n".join([
        f"{i}: {stmt['statement']}"
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_star_select_personas",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_star_personas",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_double_star_gen",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_double_star",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Optional[str]:
    """Generate a new consensus statement using preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    n_voters = len(sample_preferences[0]) if sample_preferences else 0
    rankings_summary = []
    for voter in range(n_voters):
        ranking = [sample_preferences[rank][voter] for rank in range(len(sample_preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking)}")
    
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_double_star_gen_rankings",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_double_star_rankings",
                rep=rep,
                mini_rep=mini_rep,
            ),
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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    mini_rep: int = None,
) -> Optional[str]:
    """Generate a new consensus statement using persona descriptions."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(sample_statements)
    ])
    
    # Format personas (filter to key demographic fields for all voters)
    personas_text = "\n\n".join([
        f"Voter {i+1}: {filter_persona(persona)}"
        for i, persona in enumerate(sample_personas)
    ])
    
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_double_star_gen_personas",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_double_star_personas",
                rep=rep,
                mini_rep=mini_rep,
            ),
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("new_statement")
    except Exception as e:
        logger.error(f"Generate new statement with personas failed: {e}")
        return None


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
    model: str = GENERATIVE_VOTING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
) -> Optional[str]:
    """
    Generate a bridging statement given ONLY the topic (no existing statements).
    
    This is the ChatGPT*** method - completely blind generation.
    
    Args:
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        rep: Replication number (for metadata)
    
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
            reasoning={"effort": GENERATIVE_VOTING_REASONING},
            metadata=build_api_metadata(
                phase="3_selection",
                component="gpt_triple_star_gen",
                topic=topic,
                voter_dist=voter_dist,
                alt_dist=alt_dist,
                method="chatgpt_triple_star",
                rep=rep,
            ),
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        return result.get("bridging_statement")
    except Exception as e:
        logger.error(f"Generate bridging statement (no context) failed: {e}")
        return None
