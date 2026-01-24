"""
Voter sampling modules for the sample-alt-voters experiment.

Provides two sampling strategies:
- uniform: Sample uniformly from all 815 adult personas
- clustered: Sample from ideology-based clusters (Progressive/Liberal or Conservative/Traditional)
"""

from .uniform import sample_uniform
from .clustered import sample_from_cluster

__all__ = ["sample_uniform", "sample_from_cluster"]
