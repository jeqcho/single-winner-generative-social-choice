"""
Experiment utilities module.

Shared utilities for voting experiments including:
- Voting methods (traditional and GPT-based)
- Epsilon calculation for consensus measurement
- Single-call ranking utilities
- API metadata for OpenAI dashboard tracking
"""

from .config import build_api_metadata, get_run_id, set_run_id

__all__ = ["build_api_metadata", "get_run_id", "set_run_id"]
