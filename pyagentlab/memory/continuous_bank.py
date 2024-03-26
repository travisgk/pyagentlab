"""
pyagentlab/memory/continuous_bank.py
---
this file defines a memory bank that stores Transitions
without the use of Episodes.

"""

from collections import deque
import numpy as np
from ._transition_bank import TransitionBank


class ContinuousBank(TransitionBank):
    def __init__(self, PROFILE):
        super().__init__()
        self._perspectives = deque(maxlen=PROFILE.MAX_MEM_SIZE)
        self._transitions = deque(maxlen=PROFILE.MAX_MEM_SIZE)

    # appends to the given <transitions_list> and <perspectives_list>
    # with <minibatch_size> perspectives and Transitions.
    def sample_transitions(
        self,
        transitions_list,
        perspectives_list,
        minibatch_size,
        sample_entire_episodes=False,
    ):
        transition_indices = np.random.choice(
            len(self._transitions), minibatch_size, replace=False
        )
        for index in transition_indices:
            transitions_list.append(self._transitions[index])
            perspectives_list.append(self._perspectives[index])

    def store_transition(self, transition, perspective):
        self._transitions.append(transition)
        self._perspectives.append(perspective)

    def length(self):
        return len(self._transitions)
