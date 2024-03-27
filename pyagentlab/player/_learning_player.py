"""
pyagentlab/player/_learning_player.py
---
this file, exclusive to the 'player' module, defines a class that is an agent 
which can interact with the environment and store
Transitions between states, which can be used by child classes to learn.

"""

from pyagentlab.environment.outcome import OUTCOME
from pyagentlab.environment.state import BLANK_STATE, State
from pyagentlab.memory.continuous_bank import ContinuousBank
from pyagentlab.memory.episode import Episode
from pyagentlab.memory.episode_bank import EpisodeBank
from pyagentlab.memory.transition import Transition
from ._last_prev_states import LastPrevStatesByPerspective
from .player import Player


class _LearningPlayer(Player):
    def __init__(self, PROFILE):
        super().__init__(PROFILE)
        self.can_learn = True
        self._transition_bank = (
            ContinuousBank(self.PROFILE)
            if self.PROFILE.CONTINUOUS_MEMORY
            else EpisodeBank(self.PROFILE)
        )
        self.reset()

    def reset(self):
        super().reset()
        if not self.PROFILE.CONTINUOUS_MEMORY:
            self._active_episodes = []
        self._last_prev = LastPrevStatesByPerspective()

    # this method should be run after the environment is stepped forward.
    # the transition from <self.last_prev.states[player_num - 1]>
    # to the previous state <state> will be stored.
    def process_step_and_learn(
        self, player_num, state, conv, add_fc, action, reward, done, legal
    ):
        self._allocate_new_perspective(player_num)
        index = player_num - 1
        transitions_and_legality = []

        # stores transition from the last previous state to previous state.
        if self._last_prev.states[index]:
            transitions_and_legality.append(
                self._last_prev.create_transition_last_to_prev(
                    self.PROFILE, index, state, conv, add_fc
                )
            )

        # processes transition from the previous state to terminal state.
        # since a terminal state has a Q-value of 0.0, the state is arbitrary.
        if done:
            transitions_and_legality.append(
                (
                    Transition(
                        state,
                        conv,
                        add_fc,
                        BLANK_STATE,
                        State.BLANK_CONV_OBS,
                        State.BLANK_ADD_FC_OBS,
                        action,
                        reward if legal else self.PROFILE.ILLEGAL_REWARD,
                        True,
                    ),
                    legal,
                )
            )

        # stores the Transitions in the TransitionBank.
        if self.PROFILE.CONTINUOUS_MEMORY:
            for transition, is_legal in transitions_and_legality:
                self._transition_bank.store_transition(transition, player_num)
        else:
            for transition, is_legal in transitions_and_legality:
                self._active_episodes[index].add_entry(transition, is_legal)

        # resets slot.
        if done:
            self._last_prev.reset_slot(index)

        # previous state becomes the next state.
        else:
            self._last_prev.set_slot(index, state, conv, add_fc, action, reward, legal)

    # fills lists with None in order to contain the last inputs of
    # <process_step_and_learn>.
    # creates a new episode to maintain Transitions if specified.
    def _allocate_new_perspective(self, perspective):
        super()._allocate_new_perspective(perspective)
        player_index = perspective - 1
        n_slots = len(self._last_prev.states)
        if n_slots < perspective:
            n_new_slots = perspective - n_slots
            self._last_prev.allocate_new_slots(n_new_slots)
            if not self.PROFILE.CONTINUOUS_MEMORY:
                self._active_episodes.extend([None for _ in range(n_new_slots)])
        if (
            not self.PROFILE.CONTINUOUS_MEMORY
            and not self._active_episodes[player_index]
        ):
            self._active_episodes[player_index] = Episode(perspective)

    # this method should be run after the end of an episode.
    # the transition from <self.last_prev_state>
    # to an arbitrary terminal_state will be stored.
    # <player_outcomes> come from the Environment class
    # which are retrieved inside the <play_episode()> function.
    # returns a list of total scores for each contained episode by perspective.
    def finalize_episodes(self, player_outcomes, win_ranks):
        scores = [None for _ in range(len(self._active_episodes))]
        active_slots = self._last_prev.get_active_slots()

        for player_index in active_slots:
            if player_outcomes[player_index] != OUTCOME.INTERRUPTED:
                # since a terminal state has a Q-value of 0.0,
                # the given next state and observations are arbitrary.
                transition, legal = self._last_prev.create_transition_last_to_prev(
                    self.PROFILE,
                    player_index,
                    BLANK_STATE,
                    State.BLANK_CONV_OBS,
                    State.BLANK_ADD_FC_OBS,
                    True,
                )

                # stores this last transition in TransitionBank.
                if self.PROFILE.CONTINUOUS_MEMORY:
                    self._transition_bank.store_transition(transition, player_index + 1)
                else:
                    self._active_episodes[player_index].add_entry(transition, legal)
            self._last_prev.reset_slot(player_index)

        # finalizes the episodes (if being used) and stores them in memory.
        if not self.PROFILE.CONTINUOUS_MEMORY:
            active_episodes = [ep for ep in self._active_episodes if ep]
            worst_rank = max([x for x in win_ranks if x])
            for i, active_ep in enumerate(active_episodes):
                scores[i] = active_ep.finalize(
                    self.PROFILE, player_outcomes[i], win_ranks[i], worst_rank
                )
                self._transition_bank.store_finalized_episode(active_ep)

        self.reset()
        return scores
