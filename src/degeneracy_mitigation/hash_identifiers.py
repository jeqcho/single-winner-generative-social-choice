"""
Hash identifiers for statements.

Generate deterministic 4-character hashes from statement IDs to break
the association between statement indices (0-99) and rank positions.
"""

import hashlib
from .config import SAFE_CHARS, HASH_SEED


def id_to_hash(statement_id: int, seed: int = HASH_SEED) -> str:
    """
    Generate a deterministic 4-character hash from a statement ID.
    
    Args:
        statement_id: The integer ID of the statement (0-99)
        seed: Seed for deterministic generation (default: HASH_SEED)
    
    Returns:
        A 4-character string using only unambiguous characters.
    
    Example:
        >>> id_to_hash(0, seed=42)
        'aB3x'  # (example output, actual will vary)
    """
    input_str = f"{seed}:{statement_id}"
    digest = hashlib.sha256(input_str.encode()).digest()
    
    # Convert first 4 bytes to indices into SAFE_CHARS
    result = ""
    for i in range(4):
        result += SAFE_CHARS[digest[i] % len(SAFE_CHARS)]
    return result


def hash_to_id(hash_str: str, n_statements: int, seed: int = HASH_SEED) -> int:
    """
    Reverse lookup - find which statement ID produces the given hash.
    
    Args:
        hash_str: The 4-character hash to look up
        n_statements: Total number of statements to search through
        seed: Seed used for hash generation (must match id_to_hash)
    
    Returns:
        The statement ID that produces the given hash.
    
    Raises:
        ValueError: If no statement ID produces the given hash.
    """
    for i in range(n_statements):
        if id_to_hash(i, seed) == hash_str:
            return i
    raise ValueError(f"Hash '{hash_str}' not found in {n_statements} statements with seed {seed}")


def generate_all_hashes(n_statements: int, seed: int = HASH_SEED) -> dict[int, str]:
    """
    Generate hashes for all statement IDs.
    
    Args:
        n_statements: Number of statements
        seed: Seed for hash generation
    
    Returns:
        Dictionary mapping statement ID to hash string.
    """
    return {i: id_to_hash(i, seed) for i in range(n_statements)}


def build_hash_lookup(n_statements: int, seed: int = HASH_SEED) -> dict[str, int]:
    """
    Build a reverse lookup table from hashes to IDs.
    
    More efficient than calling hash_to_id repeatedly.
    
    Args:
        n_statements: Number of statements
        seed: Seed for hash generation
    
    Returns:
        Dictionary mapping hash string to statement ID.
    """
    return {id_to_hash(i, seed): i for i in range(n_statements)}


def validate_hash(hash_str: str, valid_hashes: set[str]) -> bool:
    """
    Check if a hash string is in the set of valid hashes.
    
    Args:
        hash_str: The hash to validate
        valid_hashes: Set of valid hash strings
    
    Returns:
        True if hash is valid, False otherwise.
    """
    return hash_str in valid_hashes


if __name__ == "__main__":
    # Quick test
    print("Testing hash generation...")
    n = 100
    hashes = generate_all_hashes(n)
    
    # Check uniqueness
    unique_hashes = set(hashes.values())
    print(f"Generated {len(hashes)} hashes, {len(unique_hashes)} unique")
    assert len(unique_hashes) == n, "Collision detected!"
    
    # Test reverse lookup
    lookup = build_hash_lookup(n)
    for i in range(n):
        h = id_to_hash(i)
        recovered = lookup[h]
        assert recovered == i, f"Reverse lookup failed: {i} -> {h} -> {recovered}"
    
    print("All tests passed!")
    print(f"\nFirst 10 hashes:")
    for i in range(10):
        print(f"  {i}: {id_to_hash(i)}")
