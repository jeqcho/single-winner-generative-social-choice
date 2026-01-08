"""
Configuration constants for the full experiment.
"""

import time
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration
# =============================================================================
MODEL = "gpt-5-nano"
BRIDGING_MODEL = "gpt-5.2"     # Model for generating bridging statements
FILTERING_MODEL = "gpt-5.2"   # Model for filtering/clustering statements
TEMPERATURE = 1  # gpt-5-nano requires temperature=1

# =============================================================================
# Experiment Parameters
# =============================================================================
N_PERSONAS = 100           # Number of personas to use
N_STATEMENTS = 100         # Number of statements to sample
N_SAMPLE_PERSONAS = 20     # Number of personas to sample for voting
N_STATEMENT_REPS = 5       # Number of statement sampling repetitions (outer loop)
N_PERSONA_SAMPLES = 10     # Number of persona sampling repetitions (inner loop)
BASE_SEED = 42             # Base random seed

# Test mode parameters (reduced for faster testing)
N_STATEMENT_REPS_TEST = 2  # 2 reps instead of 5
N_PERSONA_SAMPLES_TEST = 3 # 3 samples instead of 10

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_experiment"
STATEMENTS_DIR = DATA_DIR / "large_scale" / "prod" / "statements"

# =============================================================================
# Topic Slugs (all 13 topics)
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

# Test topics (2 for quick testing)
TEST_TOPICS = [
    "what-principles-should-guide-immigration-policy-an",
    "what-are-the-best-policies-to-prevent-littering-in",
]

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

# Display names for topics (for plot titles and reports)
TOPIC_DISPLAY_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "Public Trust",
    "what-are-the-best-policies-to-prevent-littering-in": "Littering",
    "what-are-your-thoughts-on-the-way-university-campu": "Campus Speech",
    "what-balance-should-be-struck-between-environmenta": "Environment",
    "what-balance-should-exist-between-gun-safety-laws-": "Gun Safety",
    "what-limits-if-any-should-exist-on-free-speech-reg": "Free Speech",
    "what-principles-should-guide-immigration-policy-an": "Immigration",
    "what-reforms-if-any-should-replace-or-modify-the-e": "Electoral College",
    "what-responsibilities-should-tech-companies-have-w": "Tech Privacy",
    "what-role-should-artificial-intelligence-play-in-s": "AI",
    "what-role-should-the-government-play-in-ensuring-u": "Healthcare",
    "what-should-guide-laws-concerning-abortion": "Abortion",
    "what-strategies-should-guide-policing-to-address-b": "Policing",
}

# =============================================================================
# Topic to full question mapping
# =============================================================================
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

# =============================================================================
# Voting Methods
# =============================================================================
VOTING_METHODS = [
    "schulze",
    "veto_by_consumption",
    "borda",
    "irv",
    "plurality",
    "chatgpt",
    "chatgpt_with_rankings",
    "chatgpt_with_personas",
]

# =============================================================================
# Ablation Configurations
# =============================================================================
ABLATION_FULL = "full"              # Full pipeline
ABLATION_NO_BRIDGING = "no_bridging"  # Skip bridging generation
ABLATION_NO_FILTERING = "no_filtering"  # Skip filtering

ABLATIONS = [ABLATION_FULL, ABLATION_NO_BRIDGING, ABLATION_NO_FILTERING]

# =============================================================================
# Parallelization
# =============================================================================
MAX_WORKERS = 50  # Maximum parallel API calls


# =============================================================================
# API Timing Tracker
# =============================================================================
class APITimer:
    """
    Thread-safe tracker for API call timing.
    Logs average time periodically (every LOG_INTERVAL calls).
    """
    LOG_INTERVAL = 500  # Log average every N calls
    
    def __init__(self):
        self._times = []
        self._lock = threading.Lock()
        self._call_count = 0
    
    def record(self, duration: float) -> None:
        """Record an API call duration and maybe log stats."""
        with self._lock:
            self._times.append(duration)
            self._call_count += 1
            
            if self._call_count % self.LOG_INTERVAL == 0:
                avg = sum(self._times) / len(self._times)
                total = len(self._times)
                logger.info(
                    f"[API Stats] {total} calls, avg {avg:.2f}s/call, "
                    f"last {self.LOG_INTERVAL} avg {sum(self._times[-self.LOG_INTERVAL:]) / self.LOG_INTERVAL:.2f}s"
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
                "min": min(self._times),
                "max": max(self._times),
            }
    
    def reset(self) -> None:
        """Reset all tracked times."""
        with self._lock:
            self._times = []
            self._call_count = 0


# Global timer instance
api_timer = APITimer()

