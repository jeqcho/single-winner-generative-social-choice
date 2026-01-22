"""
Configuration constants for the sample-alt-voters experiment.

This experiment explores 4 alternative distributions × 2 voter distributions
across 2 contentious topics (abortion and electoral college).
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration
# =============================================================================
MODEL = "gpt-5.2"  # Use GPT-5.2 for all API calls
TEMPERATURE = 1.0

# =============================================================================
# Experiment Parameters
# =============================================================================
N_GLOBAL_PERSONAS = 815    # Total adult personas in the global population
N_GLOBAL_ALT1 = 815        # Pre-generated Alt1 statements (one per persona)
N_GLOBAL_ALT4 = 815        # Pre-generated Alt4 statements

N_ALTERNATIVES = 100       # Number of alternatives per rep
N_VOTERS = 100             # Number of voters per rep
K_SAMPLE = 20              # Voters per mini-rep
P_SAMPLE = 20              # Alternatives per mini-rep
N_SAMPLES_PER_REP = 5      # Mini-reps per rep (samples for voting evaluation)

N_REPS = 10                # Number of replications
N_CLUSTERS = 10            # Number of persona clusters (K=10)

BASE_SEED = 42             # Base random seed

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input data paths (existing data)
PERSONAS_PATH = DATA_DIR / "personas" / "prod" / "adult.json"  # 815 adult personas
STATEMENTS_ADULT_DIR = DATA_DIR / "large_scale" / "prod" / "statements_adult"  # Adult-only statements
CLUSTER_ASSIGNMENTS_PATH = DATA_DIR / "persona_embeddings_adult" / "persona_clusters.json"

# Output paths for generated/sampled data
OUTPUT_BASE_DIR = DATA_DIR / "sample-alt-voters"
SAMPLED_STATEMENTS_DIR = OUTPUT_BASE_DIR / "sampled-statements"
SAMPLED_CONTEXT_DIR = OUTPUT_BASE_DIR / "sampled-context"
SAMPLED_PERSONAS_DIR = OUTPUT_BASE_DIR / "sampled-personas"

# Results output
RESULTS_DIR = PROJECT_ROOT / "outputs" / "sample_alt_voters"

# =============================================================================
# Topic Configuration
# =============================================================================
# We focus on two contentious topics
TOPICS = [
    "what-should-guide-laws-concerning-abortion",
    "what-reforms-if-any-should-replace-or-modify-the-e",
]

TOPIC_QUESTIONS = {
    "what-should-guide-laws-concerning-abortion": 
        "What should guide laws concerning abortion?",
    "what-reforms-if-any-should-replace-or-modify-the-e":
        "What reforms, if any, should replace or modify the electoral college?",
}

TOPIC_SHORT_NAMES = {
    "what-should-guide-laws-concerning-abortion": "abortion",
    "what-reforms-if-any-should-replace-or-modify-the-e": "electoral",
}

# =============================================================================
# Alternative Distributions
# =============================================================================
ALT_DISTRIBUTIONS = [
    "persona_no_context",      # Alt1: persona, NO context (pre-generated)
    "persona_context",         # Alt2: persona + context (Ben's bridging, per-rep)
    "no_persona_context",      # Alt3: NO persona + context (verbalized, per-rep)
    "no_persona_no_context",   # Alt4: NO persona, NO context (verbalized, pre-generated)
]

# Pre-generated distributions (can be loaded once)
PRE_GENERATED_ALTS = ["persona_no_context", "no_persona_no_context"]

# Per-rep generated distributions (depend on sampled context)
PER_REP_ALTS = ["persona_context", "no_persona_context"]

# =============================================================================
# Voter Distributions
# =============================================================================
VOTER_DISTRIBUTIONS = ["uniform", "clustered"]

# =============================================================================
# Parallelization
# =============================================================================
MAX_WORKERS = 50  # Maximum parallel API calls for statement generation

# =============================================================================
# API Timing Tracker
# =============================================================================
class APITimer:
    """Thread-safe tracker for API call timing."""
    LOG_INTERVAL = 100
    
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
