"""CS378 Assignment 4: research agents and KV-cache policy optimization.

from agentkv import score
from agentkv.policy import Policy
"""

from .policy import Policy


def score(*args, **kwargs):
    """See agentkv.score.score."""
    from .score import score as _score

    return _score(*args, **kwargs)


__all__ = ["Policy", "score"]
