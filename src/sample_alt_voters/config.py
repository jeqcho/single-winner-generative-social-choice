"""
Configuration constants for the sample-alt-voters experiment.

This experiment explores 4 alternative distributions × 2 voter distributions
across 2 contentious topics (abortion and electoral college).

Phase 2 uses A*-low iterative ranking for preference building to avoid
the 81% degeneracy problem from single-call ranking.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration
# =============================================================================
# Use gpt-5-mini for Phase 2 (cost-effective, validated with A*-low)
MODEL = "gpt-5-mini"
TEMPERATURE = 1.0

# Reasoning effort for A*-low iterative ranking
REASONING_EFFORT = "low"

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

# Phase 2 replication structure
N_REPS_UNIFORM = 10        # Replications for uniform voter distribution
N_REPS_CLUSTERED = 2       # Replications for clustered (1 per ideology)
N_REPS = 10                # Default replications (for backward compatibility)

# Ideology clusters
IDEOLOGY_CLUSTERS = ["progressive_liberal", "conservative_traditional"]
N_CLUSTERS = 2             # Number of ideology clusters

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

# Output paths for generated/sampled data (Phase 1)
OUTPUT_BASE_DIR = DATA_DIR / "sample-alt-voters"
SAMPLED_STATEMENTS_DIR = OUTPUT_BASE_DIR / "sampled-statements"
SAMPLED_CONTEXT_DIR = OUTPUT_BASE_DIR / "sampled-context"
SAMPLED_PERSONAS_DIR = OUTPUT_BASE_DIR / "sampled-personas"
IDEOLOGY_CLUSTERS_PATH = OUTPUT_BASE_DIR / "ideology_clusters.json"

# Results output (Phase 2)
RESULTS_DIR = PROJECT_ROOT / "outputs" / "sample_alt_voters"
PHASE2_DATA_DIR = RESULTS_DIR / "data"
PHASE2_FIGURES_DIR = RESULTS_DIR / "figures"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

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
