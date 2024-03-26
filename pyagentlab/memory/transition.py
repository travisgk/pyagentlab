"""
pyagentlab/memory/transition.py
---
this file defines a Transition between one previous state and the next.
it stores the State in order to access information like legal moves, if needed.

"""


class Transition:
    def __init__(
        self,
        prev_state,
        prev_conv_obs,
        prev_add_fc_obs,
        next_state,
        next_conv_obs,
        next_add_fc_obs,
        action,
        reward,
        next_is_done,
    ):
        self.prev_state = prev_state
        self.prev_conv_obs = prev_conv_obs
        self.prev_add_fc_obs = prev_add_fc_obs
        self.next_state = next_state
        self.next_conv_obs = next_conv_obs
        self.next_add_fc_obs = next_add_fc_obs
        self.action = action
        self.reward = reward
        self.next_is_done = next_is_done
