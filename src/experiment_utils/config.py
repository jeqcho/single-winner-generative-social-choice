"""
Configuration constants for the sampling experiment.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Model Constants (Single Source of Truth)
# =============================================================================
# Note on reasoning effort support:
#   gpt-5-mini: minimal, low, medium, high (NOT none or xhigh)
#   gpt-5.2:    none, low, medium, high, xhigh (NOT minimal)

# STATEMENT_MODEL: Used for statement/alternative generation (Phase 1)
#   - Alt1, Alt2, Alt3, Alt4 statement generation
STATEMENT_MODEL = "gpt-5-mini"
STATEMENT_REASONING = "minimal"

# GENERATIVE_VOTING_MODEL: Used for GPT-based voting methods (Phase 3)
#   - GPT/GPT* selection
#   - GPT**/GPT*** new statement generation
GENERATIVE_VOTING_MODEL = "gpt-5.2"
GENERATIVE_VOTING_REASONING = "none"

# RANKING_MODEL: Used for all preference/ranking tasks
#   - Iterative ranking (Phase 2 preference building)
#   - Epsilon via insertion (inserting new statements into rankings)
RANKING_MODEL = "gpt-5-mini"
RANKING_REASONING = "low"

# =============================================================================
# API Configuration
# =============================================================================
TEMPERATURE = 1.0

# =============================================================================
# Experiment Parameters
# =============================================================================
N_VOTER_POOL = 100         # Number of personas in voter pool
N_ALT_POOL = 100           # Number of alternatives in alternative pool
N_REPS = 10                # Number of replications

# Sampling parameters
K_VALUES = [10, 20, 50, 100]    # Number of voters to sample
P_VALUES = [10, 20, 50, 100]    # Number of alternatives to sample
N_SAMPLES_PER_KP = 3           # Number of samples per (K, P) configuration

# Test mode parameters
N_REPS_TEST = 2            # Reduced reps for testing

BASE_SEED = 42             # Base random seed

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATEMENTS_DIR = DATA_DIR / "large_scale" / "prod" / "statements"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sampling_experiment"

# =============================================================================
# Topic Configuration
# =============================================================================
ALL_TOPICS = [
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

# Test topic (public trust only)
TEST_TOPIC = "how-should-we-increase-the-general-publics-trust-i"

# Topic slug to full question mapping
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

# Short names for topics (for filenames)
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
# Voting Methods
# =============================================================================
TRADITIONAL_METHODS = [
    "schulze",
    "borda",
    "irv",
    "plurality",
    "veto_by_consumption",
]

CHATGPT_METHODS = [
    "chatgpt",
    "chatgpt_rankings",
    "chatgpt_personas",
]

CHATGPT_STAR_METHODS = [
    "chatgpt_star",
    "chatgpt_star_rankings",
    "chatgpt_star_personas",
]

CHATGPT_DOUBLE_STAR_METHODS = [
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
]

ALL_METHODS = (
    TRADITIONAL_METHODS + 
    CHATGPT_METHODS + 
    CHATGPT_STAR_METHODS + 
    CHATGPT_DOUBLE_STAR_METHODS
)

# =============================================================================
# Parallelization
# =============================================================================
MAX_WORKERS = 100  # Maximum parallel API calls


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


# =============================================================================
# API Metadata for OpenAI Dashboard Tracking
# =============================================================================
# Global run ID - set once at pipeline start
_RUN_ID: str = None


def get_run_id() -> str:
    """Get or generate the run ID for this pipeline execution."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_ID


def set_run_id(run_id: str) -> None:
    """Explicitly set the run ID (useful for resuming runs)."""
    global _RUN_ID
    _RUN_ID = run_id


def build_api_metadata(
    phase: str,
    component: str,
    topic: str = None,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    mini_rep: int = None,
    voter_idx: int = None,
    round_num: int = None,
) -> dict:
    """
    Build metadata dict for OpenAI API calls.
    
    Metadata is attached to responses.create() calls and visible in the OpenAI
    dashboard for filtering and cost tracking.
    
    Args:
        phase: Experiment phase ("1_statement_gen", "2_preference", "3_selection", "4_insertion")
        component: Specific component name (code-level identifier)
        topic: Topic slug
        voter_dist: Voter distribution ("uniform", "clustered")
        alt_dist: Alternative distribution type
        method: Voting method name (Phase 3/4)
        rep: Replication number
        mini_rep: Mini-rep index within a rep (Phase 3)
        voter_idx: Voter index (for ranking/insertion)
        round_num: Round number (for iterative ranking)
    
    Returns:
        Dict with up to 12 key-value pairs for OpenAI metadata parameter.
        Keys max 64 chars, values max 512 chars.
    """
    metadata = {
        "project": "gsc_single_winner",
        "run_id": get_run_id(),
        "phase": phase[:64],
        "component": component[:64],
    }
    
    # Add contextual fields if provided
    if topic:
        metadata["topic"] = topic[:64]
    if voter_dist:
        metadata["voter_dist"] = voter_dist[:64]
    if alt_dist:
        metadata["alt_dist"] = alt_dist[:64]
    if method:
        metadata["method"] = method[:64]
    if rep is not None:
        metadata["rep"] = str(rep)
    if mini_rep is not None:
        metadata["mini_rep"] = str(mini_rep)
    if voter_idx is not None:
        metadata["voter_idx"] = str(voter_idx)
    if round_num is not None:
        metadata["round"] = str(round_num)
    
    return metadata
