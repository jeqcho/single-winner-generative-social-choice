"""
Degeneracy detection and validation for preference rankings.

Detects sequential/reverse patterns (degenerate rankings) and validates
that outputs meet structural requirements (correct counts, no duplicates, valid hashes).
"""

import logging
from typing import Callable, Any

from .config import MAX_RETRIES, K_TOP_BOTTOM

logger = logging.getLogger(__name__)


def is_degenerate(ranking: list[str], presentation_order: list[str]) -> bool:
    """
    Check if ranking matches presentation order (sequential) or reverse.
    
    A degenerate ranking indicates the model took a shortcut instead of
    genuinely evaluating preferences.
    
    Args:
        ranking: The output ranking (list of hash strings)
        presentation_order: The order statements were presented to the model
    
    Returns:
        True if ranking matches presentation order or its reverse.
    """
    if not ranking or not presentation_order:
        return False
    
    # Check if ranking matches presentation order exactly
    if ranking == presentation_order:
        return True
    
    # Check if ranking matches reverse of presentation order
    if ranking == presentation_order[::-1]:
        return True
    
    return False


def is_partial_degenerate(
    top_k: list[str], 
    bottom_k: list[str], 
    presentation_order: list[str]
) -> bool:
    """
    Check if top-K/bottom-K selection is degenerate.
    
    Degenerate if:
    - top_k matches first K items of presentation_order
    - bottom_k matches last K items of presentation_order
    OR the reverse pattern.
    
    Args:
        top_k: Selected top K items
        bottom_k: Selected bottom K items
        presentation_order: The order statements were presented
    
    Returns:
        True if selection appears degenerate.
    """
    k = len(top_k)
    if k == 0:
        return False
    
    # Forward pattern: top matches first K, bottom matches last K
    first_k = presentation_order[:k]
    last_k = presentation_order[-k:]
    
    if top_k == first_k and bottom_k == last_k:
        return True
    
    # Reverse pattern: top matches last K (reversed), bottom matches first K (reversed)
    if top_k == last_k[::-1] and bottom_k == first_k[::-1]:
        return True
    
    return False


def validate_top_bottom_k(
    top_k: list[str], 
    bottom_k: list[str], 
    valid_hashes: set[str],
    k: int = K_TOP_BOTTOM
) -> tuple[bool, str]:
    """
    Validate top-K/bottom-K output from a round.
    
    Checks:
    1. top_k has exactly k items
    2. bottom_k has exactly k items
    3. No duplicates within top_k
    4. No duplicates within bottom_k
    5. No overlap between top_k and bottom_k
    6. All hashes exist in valid_hashes (statements presented this round)
    
    Args:
        top_k: List of top K hash strings
        bottom_k: List of bottom K hash strings
        valid_hashes: Set of valid hash strings for this round
        k: Expected number of items (default: K_TOP_BOTTOM from config)
    
    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    # Check lengths
    if len(top_k) != k:
        return False, f"top_k has {len(top_k)} items, expected {k}"
    if len(bottom_k) != k:
        return False, f"bottom_k has {len(bottom_k)} items, expected {k}"
    
    # Check for duplicates within each list
    if len(set(top_k)) != len(top_k):
        dups = [h for h in top_k if top_k.count(h) > 1]
        return False, f"top_k contains duplicates: {set(dups)}"
    if len(set(bottom_k)) != len(bottom_k):
        dups = [h for h in bottom_k if bottom_k.count(h) > 1]
        return False, f"bottom_k contains duplicates: {set(dups)}"
    
    # Check for overlap
    overlap = set(top_k) & set(bottom_k)
    if overlap:
        return False, f"top_k and bottom_k overlap: {overlap}"
    
    # Check all hashes are valid
    invalid_top = [h for h in top_k if h not in valid_hashes]
    if invalid_top:
        return False, f"top_k contains invalid hashes: {invalid_top}"
    invalid_bottom = [h for h in bottom_k if h not in valid_hashes]
    if invalid_bottom:
        return False, f"bottom_k contains invalid hashes: {invalid_bottom}"
    
    return True, ""


def validate_final_ranking(ranking: list[str], valid_hashes: set[str]) -> tuple[bool, str]:
    """
    Validate final round ranking (all remaining statements).
    
    Checks:
    1. Correct number of items
    2. No duplicates
    3. All hashes exist in valid_hashes
    4. Contains ALL valid_hashes (complete ranking)
    
    Args:
        ranking: List of hash strings representing the ranking
        valid_hashes: Set of all valid hash strings that should be ranked
    
    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    expected_count = len(valid_hashes)
    
    if len(ranking) != expected_count:
        return False, f"ranking has {len(ranking)} items, expected {expected_count}"
    
    if len(set(ranking)) != len(ranking):
        dups = [h for h in ranking if ranking.count(h) > 1]
        return False, f"ranking contains duplicates: {set(dups)}"
    
    ranking_set = set(ranking)
    if ranking_set != valid_hashes:
        missing = valid_hashes - ranking_set
        extra = ranking_set - valid_hashes
        return False, f"missing: {missing}, extra/invalid: {extra}"
    
    return True, ""


def validate_scores(
    scores: dict[str, float], 
    valid_hashes: set[str]
) -> tuple[bool, str]:
    """
    Validate score output from scoring approach.
    
    Checks:
    1. All valid hashes have scores
    2. No extra/invalid hashes
    3. Scores are within valid range (-100 to +100)
    
    Note: Does NOT check for duplicate scores (handled separately).
    
    Args:
        scores: Dictionary mapping hash to score
        valid_hashes: Set of hashes that should have scores
    
    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    score_hashes = set(scores.keys())
    
    if score_hashes != valid_hashes:
        missing = valid_hashes - score_hashes
        extra = score_hashes - valid_hashes
        return False, f"missing: {missing}, extra/invalid: {extra}"
    
    # Check score ranges
    for h, score in scores.items():
        if not isinstance(score, (int, float)):
            return False, f"score for {h} is not a number: {score}"
        if score < -100 or score > 100:
            return False, f"score for {h} is out of range: {score}"
    
    return True, ""


class RetryResult:
    """Result of a retry-enabled operation."""
    
    def __init__(
        self,
        result: Any,
        retry_count: int,
        is_valid: bool,
        is_degenerate: bool = False,
        error_messages: list[str] = None
    ):
        self.result = result
        self.retry_count = retry_count
        self.is_valid = is_valid
        self.is_degenerate = is_degenerate
        self.error_messages = error_messages or []


def with_retry(
    func: Callable,
    validator: Callable[[Any], tuple[bool, str]],
    degeneracy_checker: Callable[[Any], bool] = None,
    max_retries: int = MAX_RETRIES,
    **func_kwargs
) -> RetryResult:
    """
    Execute a function with retry logic on validation failure or degeneracy.
    
    Args:
        func: The function to execute (e.g., API call)
        validator: Function that validates output, returns (is_valid, error_msg)
        degeneracy_checker: Optional function to check for degeneracy
        max_retries: Maximum number of retry attempts
        **func_kwargs: Arguments to pass to func
    
    Returns:
        RetryResult with the final output and metadata.
    """
    error_messages = []
    
    for attempt in range(max_retries + 1):
        try:
            result = func(**func_kwargs)
            
            # Validate structural correctness
            is_valid, error_msg = validator(result)
            if not is_valid:
                error_messages.append(f"Attempt {attempt + 1}: Validation failed - {error_msg}")
                logger.warning(f"Validation failed on attempt {attempt + 1}/{max_retries + 1}: {error_msg}")
                continue
            
            # Check for degeneracy if checker provided
            is_degen = False
            if degeneracy_checker:
                is_degen = degeneracy_checker(result)
                if is_degen:
                    error_messages.append(f"Attempt {attempt + 1}: Degenerate output detected")
                    logger.warning(f"Degenerate output on attempt {attempt + 1}/{max_retries + 1}")
                    continue
            
            # Success!
            return RetryResult(
                result=result,
                retry_count=attempt,
                is_valid=True,
                is_degenerate=False,
                error_messages=error_messages
            )
            
        except Exception as e:
            error_messages.append(f"Attempt {attempt + 1}: Exception - {type(e).__name__}: {e}")
            logger.warning(f"Exception on attempt {attempt + 1}/{max_retries + 1}: {e}")
    
    # All retries exhausted
    logger.error(f"All {max_retries + 1} attempts failed. Errors: {error_messages}")
    
    # Return last result (may be None if all attempts raised exceptions)
    return RetryResult(
        result=result if 'result' in dir() else None,
        retry_count=max_retries,
        is_valid=False,
        is_degenerate=degeneracy_checker(result) if degeneracy_checker and 'result' in dir() else False,
        error_messages=error_messages
    )


if __name__ == "__main__":
    # Quick tests
    print("Testing degeneracy detection...")
    
    # Test is_degenerate
    order = ["a", "b", "c", "d"]
    assert is_degenerate(["a", "b", "c", "d"], order) == True
    assert is_degenerate(["d", "c", "b", "a"], order) == True
    assert is_degenerate(["a", "c", "b", "d"], order) == False
    print("  is_degenerate: PASS")
    
    # Test is_partial_degenerate
    order = ["a", "b", "c", "d", "e", "f"]
    assert is_partial_degenerate(["a", "b"], ["e", "f"], order) == True
    assert is_partial_degenerate(["f", "e"], ["b", "a"], order) == True
    assert is_partial_degenerate(["a", "c"], ["e", "f"], order) == False
    print("  is_partial_degenerate: PASS")
    
    # Test validate_top_bottom_k
    valid = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    ok, msg = validate_top_bottom_k(["a", "b"], ["i", "j"], valid, k=2)
    assert ok, msg
    
    ok, msg = validate_top_bottom_k(["a", "a"], ["i", "j"], valid, k=2)
    assert not ok
    assert "duplicates" in msg
    print("  validate_top_bottom_k: PASS")
    
    # Test validate_final_ranking
    valid = {"a", "b", "c"}
    ok, msg = validate_final_ranking(["a", "b", "c"], valid)
    assert ok, msg
    
    ok, msg = validate_final_ranking(["a", "b"], valid)
    assert not ok
    print("  validate_final_ranking: PASS")
    
    print("\nAll tests passed!")
