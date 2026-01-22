"""
Configuration for the degeneracy mitigation test.

Tests combined mitigations to reduce preference ranking degeneracy from 81% baseline.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration
# =============================================================================
MODEL = "gpt-5-mini"  # Use GPT-5-mini (cost constraint)
TEMPERATURE = 1.0

# Reasoning effort levels to test
REASONING_EFFORTS = ["minimal", "low", "medium"]

# =============================================================================
# Test Scope
# =============================================================================
# Small scope: 1 topic, 1 rep, 100 voters × 100 statements
TEST_TOPIC = "abortion"  # Short name
TEST_REP = 0

N_VOTERS = 100
N_STATEMENTS = 100

# Iterative ranking parameters
K_TOP_BOTTOM = 10  # Top 10 and bottom 10 per round
N_ROUNDS = 5       # 5 rounds total (100 -> 80 -> 60 -> 40 -> 20)

# Retry parameters
MAX_RETRIES = 3    # Max retries on degenerate/invalid output

# Scoring dedup parameters
MAX_DEDUP_ROUNDS = 3  # Max rounds to resolve duplicate scores

# =============================================================================
# Hash Configuration
# =============================================================================
HASH_SEED = 42  # Seed for deterministic hash generation

# Exclude ambiguous characters: 0/O, 1/l/I
SAFE_CHARS = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789"

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Input data paths
PERSONAS_PATH = DATA_DIR / "personas" / "prod" / "adult.json"
SAMPLED_CONTEXT_DIR = DATA_DIR / "sample-alt-voters" / "sampled-context"
SAMPLED_STATEMENTS_DIR = DATA_DIR / "sample-alt-voters" / "sampled-statements"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "degeneracy_mitigation"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Topic Configuration
# =============================================================================
TOPIC_SLUGS = {
    "abortion": "what-should-guide-laws-concerning-abortion",
    "electoral": "what-reforms-if-any-should-replace-or-modify-the-e",
}

TOPIC_QUESTIONS = {
    "abortion": "What should guide laws concerning abortion?",
    "electoral": "What reforms, if any, should replace or modify the electoral college?",
}

# =============================================================================
# System Prompts
# =============================================================================
SYSTEM_PROMPT_TEMPLATE = """You are simulating a single, internally consistent person defined by the following persona:
{persona}

You must evaluate each statement solely through the lens of this persona's values, background, beliefs, and preferences.

Your task is to {task_description} and return valid JSON only.
Do not include explanations, commentary, or extra text."""

RANKING_TASK = "rank statements by preference"
SCORING_TASK = "score statements by preference"

# =============================================================================
# Parallelization
# =============================================================================
MAX_WORKERS = 20  # Maximum parallel API calls

# =============================================================================
# API Timing Tracker
# =============================================================================
class APITimer:
    """Thread-safe tracker for API call timing."""
    LOG_INTERVAL = 50
    
    def __init__(self):
        self._times = []
        self._lock = threading.Lock()
        self._call_count = 0
    
    def record(self, duration: float) -> None:
        """Record an API call duration."""
        with self._lock:
            self._times.append(duration)
            self._call_count += 1
            
            if self._call_count % self.LOG_INTERVAL == 0:
                avg = sum(self._times) / len(self._times)
                logger.info(
                    f"[API Stats] {len(self._times)} calls, avg {avg:.2f}s/call"
                )
    
    def get_stats(self) -> dict:
        """Get current timing statistics."""
        with self._lock:
            if not self._times:
                return {"count": 0, "avg": 0, "total": 0}
            return {
                "count": len(self._times),
                "avg": sum(self._times) / len(self._times),
                "total": sum(self._times),
            }
    
    def reset(self) -> None:
        """Reset all tracked times."""
        with self._lock:
            self._times = []
            self._call_count = 0


# Global timer instance
api_timer = APITimer()
