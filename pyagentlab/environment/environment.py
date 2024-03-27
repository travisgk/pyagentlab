"""
pyagentlab/environment/environment.py
---
this file defines a class which is used to facilitate
executing moves on a State, effectively bringing the game to life.
it maintains whose turn it is, which players are out,
the outcomes for each player, and the rank of each of these outcomes.

"""

import copy
from collections import deque
import numpy as np
from pyagentlab.constants import Const
from .outcome import OUTCOME
from .state import State


class Environment:
    _DRAW_RANK = 500000

    def __init__(self, StateClass):
        self._n_steps = 0

        # deques for each perspective with values for graphing.
        self._episode_scores_history = [
            deque(maxlen=100) for _ in range(Const.N_PLAYERS)
        ]
        self._episode_outcomes_history = [
            deque(maxlen=100) for _ in range(Const.N_PLAYERS)
        ]

        # lists containing the averages across the above deques.
        # tuples with x and y values.
        # these are for graphing (not yet implemented).
        self.avg_episode_scores = [[] for _ in range(Const.N_PLAYERS)]
        self.avg_episode_outcomes = [[] for _ in range(Const.N_PLAYERS)]

        self.StateClass = StateClass if StateClass else State
        StateClass.setup()
        self.state = self.StateClass()
        self.reset()

    # resets the environment to an initial state. returns the first state.
    def reset(self):
        self._player_num = 1
        self._player_dones = [False for _ in range(Const.N_PLAYERS)]
        self._player_outcomes = [None for _ in range(Const.N_PLAYERS)]

        self._player_win_ranks = [None for _ in range(Const.N_PLAYERS)]
        self._last_win_rank = 1
        self._last_loss_rank = 999999
        self.local_n_steps = 0
        self._outcome = None
        self.state.reset()
        return copy.copy(self.state), None

    # renders the environment. intentionally empty.
    def render(self):
        return

    # closes the environment. intentionally empty.
    def close(self):
        return

    def entirely_done(self):
        return all(done for done in self._player_dones)

    # updates an environment with a given action and returns
    # 1) the next resulting state, 2) the reward for taking that action,
    # 3) if the environment has terminated or 4) truncated, 5) info,
    # and 6) if the action taken was legal.
    # outcomes of the environment are determined here.
    def step(self, action):
        returned_state = self.state
        reward = 0.0
        trunc = None
        info = None
        legal = False
        player_index = self._player_num - 1
        if not self.entirely_done():
            reward, self._player_dones[player_index], outcome, legal = (
                self.state.take_action(action, self._player_num, self.local_n_steps)
            )

            self._player_outcomes[player_index] = outcome
            if outcome == OUTCOME.WIN:
                self._player_win_ranks[player_index] = self._last_win_rank
                self._last_win_rank += 1
            elif outcome == OUTCOME.LOSS:
                self._player_win_ranks[player_index] = self._last_loss_rank
                self._last_loss_rank -= 1
            elif outcome == OUTCOME.DRAW:
                self._player_win_ranks[player_index] = Environment._DRAW_RANK
            self._determine_final_outcomes(outcome, legal)

            if legal or (not legal and Const.ILLEGAL_MOVE_STEPS_ENV):
                self._step_player_num()
                self._n_steps += 1
                self.local_n_steps += 1

        return (
            copy.copy(returned_state),
            reward,
            self._player_dones[player_index],
            trunc,
            info,
            legal,
        )

    # sets the outcomes for players with ambiguous outcomes,
    # once the environment has been brought to a complete end by <outcome>
    # or possibly an illegal move.
    def _determine_final_outcomes(self, player_outcome, legal):
        completely_done = False
        outs = self._player_outcomes
        uncertain_outs = [None, OUTCOME.DRAW]

        # determines the outcome value for all uncertain outcome values.
        if any([out in uncertain_outs for out in self._player_outcomes]):
            global_outcome = OUTCOME.LOSS

            if not legal and Const.ILLEGAL_ENDS_ENTIRE_ENV:
                completely_done = True
                global_outcome = (
                    OUTCOME.WIN if Const.WINS_BY_FORFEIT else OUTCOME.INTERRUPTED
                )
            elif player_outcome == OUTCOME.WIN and Const.WIN_ENDS_ENTIRE_ENV:
                completely_done = True
                global_outcome = (
                    OUTCOME.LOSS if Const.LOSES_BY_ENDING_WIN else OUTCOME.INTERRUPTED
                )
            elif player_outcome == OUTCOME.LOSS and Const.LOSS_ENDS_ENTIRE_ENV:
                completely_done = True
                global_outcome = (
                    OUTCOME.WIN if Const.WINS_BY_ENDING_LOSS else OUTCOME.INTERRUPTED
                )
            elif any([out == OUTCOME.DRAW for out in outs]):
                completely_done = True
                global_outcome = OUTCOME.DRAW
            else:
                active_player_indices = [
                    i for i, done in enumerate(self._player_dones) if not done
                ]
                if all(
                    [
                        not self.state.legal_plays_remain(i + 1)
                        for i in active_player_indices
                    ]
                ):
                    completely_done = True

            # sets outcome for all uncertain outcome values.
            if completely_done:
                for i in range(Const.N_PLAYERS):
                    if self._player_outcomes[i] in uncertain_outs:
                        self._player_outcomes[i] = global_outcome

        # sets all players to done and applies particular outcome behaviors.
        if completely_done:
            self._player_dones = [True for _ in self._player_dones]

            if not any([out == OUTCOME.DRAW for out in outs]):
                if not any([out == OUTCOME.WIN for out in outs]):
                    if Const.ALL_LOSSES_IS_DRAW:
                        # there are no wins or draws in the list of outcomes.
                        n_losses = self._player_outcomes.count(OUTCOME.LOSS)
                        if n_losses > 1:
                            loss_indices = [
                                i for i, out in enumerate(outs) if out == OUTCOME.LOSS
                            ]
                            for i in loss_indices:
                                self._player_outcomes[i] = OUTCOME.DRAW
                else:
                    if Const.ALL_WINS_IS_DRAW and not any(
                        [out == OUTCOME.LOSS for out in outs]
                    ):
                        n_wins = self._player_outcomes.count(OUTCOME.WIN)
                        if n_wins > 1:
                            win_indices = [
                                i for i, out in enumerate(outs) if out == OUTCOME.WIN
                            ]
                            for i in win_indices:
                                self._player_outcomes[i] = OUTCOME.DRAW

            # finalizes player rankings.
            unranked_player_indices = [
                i for i in range(Const.N_PLAYERS) if not self._player_win_ranks[i]
            ]
            for i in unranked_player_indices:
                if self._player_outcomes[i] == OUTCOME.WIN:
                    self._player_win_ranks[i] = self._last_win_rank
                elif self._player_outcomes[i] == OUTCOME.LOSS:
                    self._player_win_ranks[i] = self._last_loss_rank
                elif self._player_outcomes[i] == OUTCOME.DRAW:
                    self._player_win_ranks[i] = Environment._DRAW_RANK

            valid_ranks = [x for x in self._player_win_ranks if x]
            unique_sorted_ranks = sorted(set(valid_ranks))
            value_to_rank = {
                value: rank + 1 for rank, value in enumerate(unique_sorted_ranks)
            }
            self._player_win_ranks = [
                value_to_rank[value] if value else None
                for value in self._player_win_ranks
            ]

    # increments <self._player_num>.
    # the value fluctuates 1 to <Const.N_PLAYERS> + 1.
    def _step_player_num(self):
        if not self.entirely_done():
            while True:
                self._player_num = self._player_num % Const.N_PLAYERS + 1
                if self._player_dones[self._player_num - 1] is False:
                    break

    def get_player_num(self):
        return self._player_num

    # returns the outcome for a particular player.
    def return_outcome(self, player_num):
        return self._player_outcomes[player_num - 1]

    # returns a list of outcomes for every player.
    def return_outcomes(self):
        return self._player_outcomes

    # returns the list of player numbers representing the rankings of who won.
    def return_win_ranks(self):
        return self._player_win_ranks

    # appends players' scores and outcomes to their respective deques
    # and calculates their averages at the current timestep <self._n_steps>.
    # <history_offset> is used to offset the player index under which
    # the information is archived; this may be desired if the players
    # given to <play_episodes()> are being rotated around.
    def append_score_and_outcome_history(self, scores, history_offset=0):
        # appends to history deques.
        for i, score in enumerate(scores):
            element = (i + history_offset) % Const.N_PLAYERS
            self._episode_scores_history[element].append(score)

        for i, outcome in enumerate(self._player_outcomes):
            element = (i + history_offset) % Const.N_PLAYERS
            self._episode_outcomes_history[element].append(self._player_outcomes)

        if self._n_steps % Const.ENV_LOG_HISTORY_INTERVAL == 0 and self._n_steps > 0:
            # appends to graphable lists.
            for i in range(Const.N_PLAYERS):
                element = (i + history_offset) % Const.N_PLAYERS
                self.avg_episode_scores[element].append(
                    (self._n_steps, np.mean(self._episode_scores_history[element]))
                )

            for i in range(Const.N_PLAYERS):
                element = (i + history_offset) % Const.N_PLAYERS
                wins = 0
                losses = 0
                draws = 0
                interrupts = 0
                forfeits = 0
                for outcome in self._episode_outcomes_history[element]:
                    if outcome == OUTCOME.WIN:
                        wins += 1
                    elif outcome == OUTCOME.LOSS:
                        losses += 1
                    elif outcome == OUTCOME.DRAW:
                        draws += 1
                    elif outcome == OUTCOME.INTERRUPTED:
                        interrupts += 1
                    elif outcome == OUTCOME.FORFEIT_BY_ILLEGAL:
                        forfeits += 1
                N_OUTCOMES = len(self._episode_outcomes_history[element])
                avg_wins = int((wins / N_OUTCOMES) * 100)
                avg_losses = int((losses / N_OUTCOMES) * 100)
                avg_draws = int((draws / N_OUTCOMES) * 100)
                avg_interrupts = int((interrupts / N_OUTCOMES) * 100)
                avg_forfeits = int((forfeits / N_OUTCOMES) * 100)

                self.avg_episode_outcomes[element].append(
                    (
                        self._n_steps,
                        {
                            "wins": avg_wins,
                            "losses": avg_losses,
                            "draws": avg_draws,
                            "interrupts": avg_interrupts,
                            "forfeits": avg_forfeits,
                        },
                    )
                )
