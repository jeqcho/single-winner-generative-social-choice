"""
Sample-Alt-Voters experiment module.

This module implements a factorial experiment comparing:
- 4 alternative (statement) distributions: Alt1-Alt4
- 2 voter distributions: uniform and clustered
- 2 topics: abortion and electoral college

Key components:
- config: Experiment configuration and paths
- alternative_generators: Four methods for generating statements
- verbalized_sampling: Utilities for parsing verbalized sampling responses
- generate_statements: CLI for pre-generating Alt1 and Alt4 statements
"""

from . import config
from . import alternative_generators
from . import verbalized_sampling

__all__ = [
    "config",
    "alternative_generators",
    "verbalized_sampling",
]
