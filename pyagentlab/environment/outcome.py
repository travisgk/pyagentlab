"""
pyagentlab/environment/outcome.py
---
this file defines an enum class that 
indicates a player's outcome in an episode.

"""

from enum import Enum


class OUTCOME(Enum):
    WIN = 1
    LOSS = -1
    DRAW = 0
    INTERRUPTED = 99
    FORFEIT_BY_ILLEGAL = -99
