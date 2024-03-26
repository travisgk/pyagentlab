"""
pyagentlab/player/player.py
---
this file defines a class that is an agent 
which can interact with the environment.

the same Player object can be used to play for multiple perspectives
in a multiplayer environment. lists will be allocated so that
a particular index will represent the data unique to a particular perspective.

the general workflow with this is:
objective State --> subjective observations 

						      |
						      V
						   Network
						      |
						      V

					subjective action --> objective action

"""

from pyagentlab.constants import CONST
from pyagentlab.environment.action.random_action import random_subjective_action


class Player:
    def __init__(self, PROFILE):
        self.PROFILE = PROFILE
        self.change_epsilon(
            self.PROFILE.EPS_START, self.PROFILE.EPS_END, self.PROFILE.EPS_DEC
        )
        self.can_learn = False

    def reset(self):
        return

    def choose_action(self, state, conv, add_fc, player_num):
        if self.PROFILE.ENFORCE_LEGALITY:
            illegal_mask = state.create_illegal_subjective_action_mask(player_num)
            subjective_action = random_subjective_action(illegal_mask)
        else:
            subjective_action = random_subjective_action()
        objective_action = state.make_action_objective(subjective_action)
        return objective_action

    # this method should be run after the environment is stepped forward.
    def process_step_and_learn(
        self, player_num, state, conv, add_fc, action, reward, done, legal
    ):
        Player._allocate_new_perspective(self, player_num)
        player_index = player_num - 1
        self._decrement_eps_method()

    # allocates storage for playing information for a particular perspective.
    def _allocate_new_perspective(self, perspective):
        return

    # this method should be run after the end of an episode.
    # returns the total scores for each episode per perspective.
    def finalize_episodes(self, player_outcomes, win_ranks):
        return [None]

    # changes epsilon values and determines the method of decrement.
    def change_epsilon(self, eps_start=-1.0, eps_min=-1.0, eps_dec=-1.0):
        if eps_start >= 0.0:
            self._epsilon = eps_start
        if eps_min >= 0.0:
            self._eps_min = eps_min
        if eps_dec >= 0.0:
            self._eps_dec = eps_dec
        if self._eps_dec < 0.5:
            self._decrement_eps_method = self._linearly_decrement_eps
        else:
            self._decrement_eps_method = self._exponentially_decrement_eps

    def _linearly_decrement_eps(self):
        self._epsilon = max(self._eps_min, self._epsilon - self._eps_dec)

    def _exponentially_decrement_eps(self):
        self._epsilon = max(self._eps_min, self._epsilon * self._eps_dec)
