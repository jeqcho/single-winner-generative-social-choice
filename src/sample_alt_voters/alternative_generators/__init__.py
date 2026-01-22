"""
Alternative generators for the sample-alt-voters experiment.

This module contains four alternative (statement) distribution methods:
- persona_no_context: Alt1 - Persona generates without seeing other statements
- persona_context: Alt2 - Persona sees 100 statements then generates (Ben's bridging)
- no_persona_context: Alt3 - No persona, sees 100 statements, uses verbalized sampling
- no_persona_no_context: Alt4 - No persona, no context, uses verbalized sampling
"""

from . import persona_no_context
from . import persona_context
from . import no_persona_context
from . import no_persona_no_context

__all__ = [
    "persona_no_context",
    "persona_context",
    "no_persona_context",
    "no_persona_no_context",
]
