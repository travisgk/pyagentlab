"""
pyagentlab/player/_last_prev
---
this file defines a class used exclusively by the LearningPlayer class
to maintain the last inputs given to the LearningPlayer object
through its <LearningPlayer.process_step_and_learn()> method,
with the last inputs for each active perspective being stored
at its own respective index.

in a multiplayer game, the immediate next state can't be used
to create transitions, as the immediate next state will have
a different player, which means the value of that particular next state
cannot be approximated.

"""

from pyagentlab.environment.state import State, BLANK_STATE
from pyagentlab.memory.transition import Transition


class LastPrevStatesByPerspective:
    def __init__(self):
        self.states = []
        self.conv_obs_list = []
        self.add_fc_obs_list = []
        self.actions = []
        self.rewards = []
        self.legals = []

    def get_active_slots(self):
        return [i for i, state in enumerate(self.states) if state]

    def reset_slot(self, player_index):
        self.states[player_index] = None
        self.conv_obs_list[player_index] = None
        self.add_fc_obs_list[player_index] = None
        self.actions[player_index] = None
        self.rewards[player_index] = None
        self.legals[player_index] = None

    def set_slot(
        self,
        player_index,
        prev_state,
        prev_conv_obs,
        prev_add_fc_obs,
        action,
        reward,
        legal,
    ):
        self.states[player_index] = prev_state
        self.conv_obs_list[player_index] = prev_conv_obs
        self.add_fc_obs_list[player_index] = prev_add_fc_obs
        self.actions[player_index] = action
        self.rewards[player_index] = reward
        self.legals[player_index] = legal

    def allocate_new_slots(self, n_new_slots):
        empty_slots = [None for _ in range(n_new_slots)]
        self.states.extend(empty_slots)
        self.conv_obs_list.extend(empty_slots)
        self.add_fc_obs_list.extend(empty_slots)
        self.actions.extend(empty_slots)
        self.rewards.extend(empty_slots)
        self.legals.extend(empty_slots)

    # returns 1) a transition from the last stored values
    # to a given <prev_state> and 2) if the action was legal.
    def create_transition_last_to_prev(
        self,
        PROFILE,
        player_index,
        prev_state,
        prev_conv_obs,
        prev_add_fc_obs,
        force_terminal=False,
    ):
        if self.legals[player_index]:
            state = prev_state
            conv_obs = prev_conv_obs
            add_fc_obs = prev_add_fc_obs
            reward = self.rewards[player_index]
        else:
            state = BLANK_STATE
            conv_obs = State.BLANK_CONV_OBS
            add_fc_obs = State.BLANK_ADD_FC_OBS
            reward = PROFILE.ILLEGAL_REWARD

        # the next state resulting from an illegal move
        # will not have their total return approximated.
        done = force_terminal or self.legals[player_index] is False

        return (
            Transition(
                self.states[player_index],
                self.conv_obs_list[player_index],
                self.add_fc_obs_list[player_index],
                state,
                conv_obs,
                add_fc_obs,
                self.actions[player_index],
                reward,
                done,
            ),
            self.legals[player_index],
        )
