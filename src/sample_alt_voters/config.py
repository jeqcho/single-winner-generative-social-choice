"""
Configuration constants for the sample-alt-voters experiment.

This experiment explores 4 alternative distributions × 2 voter distributions
across 13 topics from the Polis dataset.

Phase 2 uses A-low iterative ranking for preference building to avoid
the 81% degeneracy problem from single-call ranking.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration
# =============================================================================
# Use gpt-5-mini for Phase 2 (cost-effective, validated with A-low)
MODEL = "gpt-5-mini"
TEMPERATURE = 1.0

# Reasoning effort for A-low iterative ranking
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
# All 13 topics from the Polis dataset
TOPICS = [
    "how-should-we-increase-the-general-publics-trust-i",
    "what-are-the-best-policies-to-prevent-littering-in",
    "what-are-your-thoughts-on-the-way-university-campu",
    "what-balance-should-be-struck-between-environmenta",
    "what-balance-should-exist-between-gun-safety-laws-",
    "what-limits-if-any-should-exist-on-free-speech-reg",
    "what-principles-should-guide-immigration-policy-an",
    "what-reforms-if-any-should-replace-or-modify-the-e",
    "what-responsibilities-should-tech-companies-have-w",
    "what-role-should-artificial-intelligence-play-in-s",
    "what-role-should-the-government-play-in-ensuring-u",
    "what-should-guide-laws-concerning-abortion",
    "what-strategies-should-guide-policing-to-address-b",
]

TOPIC_QUESTIONS = {
    "how-should-we-increase-the-general-publics-trust-i": 
        "How should we increase the general public's trust in institutions?",
    "what-are-the-best-policies-to-prevent-littering-in":
        "What are the best policies to prevent littering in public spaces?",
    "what-are-your-thoughts-on-the-way-university-campu":
        "What are your thoughts on the way university campuses handle free speech?",
    "what-balance-should-be-struck-between-environmenta":
        "What balance should be struck between environmental protection and economic growth?",
    "what-balance-should-exist-between-gun-safety-laws-":
        "What balance should exist between gun safety laws and Second Amendment rights?",
    "what-limits-if-any-should-exist-on-free-speech-reg":
        "What limits, if any, should exist on free speech regarding hate speech?",
    "what-principles-should-guide-immigration-policy-an":
        "What principles should guide immigration policy and the path to citizenship?",
    "what-reforms-if-any-should-replace-or-modify-the-e":
        "What reforms, if any, should replace or modify the electoral college?",
    "what-responsibilities-should-tech-companies-have-w":
        "What responsibilities should tech companies have with user data and privacy?",
    "what-role-should-artificial-intelligence-play-in-s":
        "What role should artificial intelligence play in society?",
    "what-role-should-the-government-play-in-ensuring-u":
        "What role should the government play in ensuring universal healthcare?",
    "what-should-guide-laws-concerning-abortion": 
        "What should guide laws concerning abortion?",
    "what-strategies-should-guide-policing-to-address-b":
        "What strategies should guide policing to address both safety and civil rights?",
}

TOPIC_SHORT_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "trust",
    "what-are-the-best-policies-to-prevent-littering-in": "littering",
    "what-are-your-thoughts-on-the-way-university-campu": "campus_speech",
    "what-balance-should-be-struck-between-environmenta": "environment",
    "what-balance-should-exist-between-gun-safety-laws-": "gun_safety",
    "what-limits-if-any-should-exist-on-free-speech-reg": "free_speech",
    "what-principles-should-guide-immigration-policy-an": "immigration",
    "what-reforms-if-any-should-replace-or-modify-the-e": "electoral",
    "what-responsibilities-should-tech-companies-have-w": "tech_privacy",
    "what-role-should-artificial-intelligence-play-in-s": "ai",
    "what-role-should-the-government-play-in-ensuring-u": "healthcare",
    "what-should-guide-laws-concerning-abortion": "abortion",
    "what-strategies-should-guide-policing-to-address-b": "policing",
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
