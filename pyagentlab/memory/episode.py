"""
pyagentlab/memory/episode.py
---
this file defines a class which holds a sequence of transitions
which comprise an episode from one player's perspective.

"""

import numpy as np
from pyagentlab.environment.outcome import OUTCOME
from pyagentlab.environment.state import State
from .transition import Transition


class Episode:
    def __init__(self, perspective):
        self.perspective = perspective
        self.transitions = []
        self._legals = []

    def add_entry(self, transition, legal):
        self.transitions.append(transition)
        self._legals.append(legal)

    # returns the player's total score for an episode.
    # ---
    # sets the reward for reaching a particular outcome.
    # RANK FACTORS are used to reduce the reward for worse ranks.
    # the player with a win of the 3nd best rank will have
    # WIN VALUE multiplied by its respective RANK FACTOR twice.
    # ---
    # this method also calculates the total return for each
    # Transition if it is specified to do so by the <PROFILE>.
    def finalize(self, PROFILE, player_outcome, player_rank, worst_rank):
        if len(self.transitions) <= 0:
            return

        # illegal actions are not used when calculating the total returns
        # for all the other states.
        legal_indices = [i for i in range(len(self.transitions)) if self._legals[i]]

        if len(legal_indices) > 0:
            last = legal_indices[-1]
            if player_outcome == OUTCOME.WIN:
                self.transitions[last].reward = PROFILE.WIN_VALUE * (
                    1
                    if not player_rank
                    else PROFILE.WIN_VALUE_RANK_FACTOR ** (player_rank - 1)
                )
            elif player_outcome == OUTCOME.DRAW:
                self.transitions[last].reward = PROFILE.DRAW_VALUE
            elif player_outcome == OUTCOME.LOSS:
                self.transitions[last].reward = PROFILE.LOSS_VALUE * (
                    1
                    if not player_rank or not worst_rank
                    else PROFILE.LOSS_VALUE_RANK_FACTOR ** (worst_rank - player_rank)
                )

        # calculates score to be later returned.
        score = np.sum([self.transitions[i].reward for i in legal_indices])

        if PROFILE.USE_TOTAL_RETURNS:
            legal_indices.reverse()
            G = 0.0
            for i in legal_indices:
                G = self.transitions[i].reward + PROFILE.GAMMA * G
                self.transitions[i].reward = G
        self._legals = []
        return score

    def to_str(self):
        result = ""
        for transition in self.transitions:
            result += (
                f"\n\n\n{transition.prev_state.to_str()}"
                f"perspective: {self.perspective}\n"
                f"action: {transition.action}\n"
                f"{transition.next_state.to_str()}"
                f"reward: {transition.reward:.2f}"
            )
        return result
