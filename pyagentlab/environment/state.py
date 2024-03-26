"""
pyagentlab/environment/state.py
---
this file defines a generic State in the environment.

"""

import itertools
import numpy as np
from pyagentlab.constants import CONST, uses_conv, uses_add_fc
from .outcome import OUTCOME


def _apply_perspective(objective_number, perspective):
    if objective_number == 0:
        return 0
    return (objective_number - perspective) % CONST.N_PLAYERS + 1


class State:
    ONE_HOT_TRUE = 1
    ONE_HOT_FALSE = 0
    MEM_DTYPE = np.uint8
    BLANK_CONV_OBS = None
    BLANK_ADD_FC_OBS = None

    # intentionally blank.
    def __init__(self):
        return

    def __copy__(self):
        copy_state = State()
        return copy_state

    # sets up static aspects of the State class. intentionally blank.
    def setup():
        State.BLANK_CONV_OBS = (
            np.full(CONST.CONV_INPUT_DIMS, State.ONE_HOT_FALSE, dtype=State.MEM_DTYPE)
            if uses_conv()
            else None
        )
        State.BLANK_ADD_FC_OBS = (
            np.full(CONST.ADD_FC_INPUT_DIM, State.ONE_HOT_FALSE, dtype=State.MEM_DTYPE)
            if uses_add_fc()
            else None
        )

    def reset(self):
        return

    def to_conv_obs(self, perspective):
        return State.BLANK_CONV_OBS

    def to_add_fc_obs(self, perspective):
        return State.BLANK_ADD_FC_OBS

    # returns a flattened binary mask of <player_num>'s illegal actions.
    def create_illegal_subjective_action_mask(self, player_num):
        legal_mask = self.create_legal_subjective_action_mask(player_num)
        return ~legal_mask if legal_mask is not None else None

    # returns a flattened binary mask of <player_num>'s legal actions.
    # ---
    # masking the discrete actions en-masse is a good option if
    # only one discrete action space is in use, or if
    # multiple independent discrete action spaces are used,
    # where there are no illegal combinations.
    def create_legal_subjective_action_mask(self, player_num):
        return None

    # returns True if the action can be taken.
    def action_is_legal(self, action, player_num=None):
        return True

    # returns True if any legal plays can still be taken by <player_num>.
    def legal_plays_remain(self, player_num=None):
        return True

    # returns <action> with some reverse subjective transformation applied.
    # this is useful for taking the output action of a network
    # that's been trained on subjective observations and transforming it
    # to be used on an objective state. for example, a chess network
    # trained to see everything from the perspective of player one
    # would need its outputted action transformed to play as player two.
    def make_action_objective(self, subjective_action, player_num=None):
        return subjective_action

    # returns 1) the reward for taking the action in the state,
    # and 2) a bool indicating if the resulting state is a terminal state,
    # 3) an indicator of if the player has won/lost the game,
    # and 4) a bool indicating if the action were legal.
    def take_action(self, action, player_num=None, local_n_steps=None):
        reward = 0.0
        player_done = False
        outcome = None
        legal = True

        if not self.action_is_legal(action, player_num):
            player_done = CONST.ILLEGAL_ENDS_PLAYER
            if CONST.ILLEGAL_ENDS_PLAYER or CONST.ILLEGAL_ENDS_ENTIRE_ENV:
                outcome = (
                    OUTCOME.LOSS
                    if CONST.ILLEGAL_END_IS_LOSS
                    else OUTCOME.FORFEIT_BY_ILLEGAL
                )
            legal = False
            return reward, player_done, outcome, legal

        self._apply_action(action, player_num)

        if not self.legal_plays_remain(player_num):
            player_done = True

        # determines outcome using methods
        # that can be overridden in child classes.
        if self._win_condition(action, player_num, local_n_steps):
            player_done = True
            outcome = OUTCOME.WIN
        elif self._loss_condition(action, player_num, local_n_steps):
            player_done = True
            outcome = OUTCOME.LOSS
        elif self._draw_condition(action, player_num, local_n_steps):
            player_done = True
            outcome = OUTCOME.DRAW

        return reward, player_done, outcome, legal

    # applies a legal action to the State.
    # this method is intentionally blank and left to be manually overridden.
    def _apply_action(self, action, player_num=None):
        return

    # returns True if some winning condition has been met by playing <action>.
    # this method is intentionally blank and left to be manually overridden.
    def _win_condition(self, action, player_num=None, local_n_steps=None):
        return False

    # returns True if some losing condition has been met by playing <action>.
    # this method is intentionally blank and left to be manually overridden.
    def _loss_condition(self, action, player_num=None, local_n_steps=None):
        return False

    # returns True if some drawing condition has been met by playing <action>.
    # this method is intentionally blank and left to be manually overridden.
    def _draw_condition(self, action, player_num=None, local_n_steps=None):
        return False

    def to_str(self):
        result = ""
        return result


# used as the terminal state when creating a transition.
BLANK_STATE = State()
