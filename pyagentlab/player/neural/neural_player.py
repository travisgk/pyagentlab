"""
pyagentlab/player/neural/neural_player.py
---
this file defines a class that is a neural network agent 
which can interact with the environment 
and store Transitions between states, 
then sample these Transitions to train its networks.

"""

from collections import deque
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pyagentlab.constants import CONST, uses_conv, uses_add_fc
from pyagentlab.environment.action.action import (
    action_to_combo_num,
    subjective_action_from_network_output,
)
from pyagentlab.environment.action.random_action import random_subjective_action
from pyagentlab.player._learning_player import _LearningPlayer
from .neural_network import NeuralNetwork


class NeuralPlayer(_LearningPlayer):
    def __init__(self, PROFILE):
        super().__init__(PROFILE)
        self._device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self._q_eval = NeuralNetwork(PROFILE, "q_eval").to(self._device)
        self._q_next = (
            None
            if self.PROFILE.REPLACE_TARGET_INTERVAL == 0
            else NeuralNetwork(PROFILE, "q_next").to(self._device)
        )
        self._optimizer = optim.RMSprop(
            self._q_eval.parameters(),
            lr=self.PROFILE.LR,
            alpha=self.PROFILE.ALPHA,
            momentum=self.PROFILE.GRADIENT_MOMENTUM,
            weight_decay=self.PROFILE.WEIGHT_DECAY,
            eps=self.PROFILE.MIN_SQUARED_GRADIENT,
        )
        self._loss_history = deque(maxlen=100)
        self._n_learn_steps = 0
        self._select_t_preds_method = None
        self._select_t_targets_method = None
        self._lr_scheduler = (
            lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                factor=self.PROFILE.LR_SCHEDULER_FACTOR,
                patience=self.PROFILE.LR_SCHEDULER_PATIENCE,
                verbose=False,
            )
            if self.PROFILE.LR_SCHEDULER_FACTOR > 0.0
            else None
        )

    def choose_action(self, state, conv, add_fc, player_num):
        illegal_mask = None

        # the action is chosen by the neural network.
        if np.random.random() > self._epsilon:
            with T.no_grad():
                t_conv = (
                    T.tensor(np.expand_dims(conv, axis=0), dtype=T.float32).to(
                        self._device
                    )
                    if uses_conv()
                    else None
                )
                t_add_fc = (
                    T.tensor(np.expand_dims(add_fc, axis=0), dtype=T.float32).to(
                        self._device
                    )
                    if uses_add_fc()
                    else None
                )
                t_outputs = self._q_eval.forward(t_conv, t_add_fc)

            if self.PROFILE.ENFORCE_LEGALITY:
                illegal_mask = state.create_illegal_subjective_action_mask(player_num)
            subjective_action = subjective_action_from_network_output(
                t_outputs.cpu().numpy()[0], illegal_mask
            )

        # the action is random.
        else:
            if self.PROFILE.ENFORCE_LEGALITY_ON_RANDOM:
                illegal_mask = state.create_illegal_subjective_action_mask(player_num)
            subjective_action = random_subjective_action(illegal_mask)

        objective_action = state.make_action_objective(subjective_action)
        return objective_action

    # this method should be run after the environment is stepped forward.
    def process_step_and_learn(
        self, player_num, state, conv, add_fc, action, reward, done, legal
    ):
        super().process_step_and_learn(
            player_num, state, conv, add_fc, action, reward, done, legal
        )
        self._train()

    # this method should be run after the end of an episode.
    # it will return the scores for each active episode inside the player.
    def finalize_episodes(self, player_outcomes, win_ranks):
        scores = super().finalize_episodes(player_outcomes, win_ranks)
        if not self.PROFILE.CONTINUOUS_MEMORY:
            self._replace_target_network()
        return scores

    def change_lr(self, lr):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def get_current_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    def calc_avg_loss(self, sample_size=10):
        return (
            np.mean(list(self._loss_history)[-sample_size:])
            if len(self._loss_history) >= 1
            else 0.0
        )

    def _replace_target_network(self):
        if 0 < self.PROFILE.REPLACE_TARGET_INTERVAL <= self._n_learn_steps:
            self._q_next.load_state_dict(self._q_eval.state_dict())
            self._n_learn_steps = 0

    def save_checkpoints(self, addendum=""):
        self._q_eval.save_checkpoint(addendum)
        if self._q_next:
            self._q_next.save_checkpoint(addendum)

    def load_checkpoints(self, addendum):
        self._q_eval.load_checkpoint(addendum)
        if self._q_next:
            self._q_next.load_checkpoint(addendum)

    def set_select_t_preds_method(self, method):
        self._select_t_preds_method = method

    def set_select_t_targets_method(self, method):
        self._select_t_targets_method = method

    def _train(self, sample_entire_episodes=False, scramble_samples=False):
        if self._transition_bank.length() < self.PROFILE.STARTING_MEM_COUNT:
            return
        self._optimizer.zero_grad()

        transitions = []
        perspectives = []
        self._transition_bank.sample_transitions(
            transitions,
            perspectives,
            self.PROFILE.MINIBATCH_SIZE,
            sample_entire_episodes,
        )

        if scramble_samples:
            indices = np.random.permutation(self.PROFILE.MINIBATCH_SIZE)
            transitions = [transitions[i] for i in indices]
            perspectives = [perspectives[i] for i in perspectives]

        # converts numpy arrays to tensors and puts them on the device.
        t_prev_conv = (
            T.tensor(
                np.array([t.prev_conv_obs for t in transitions]), dtype=T.float32
            ).to(self._device)
            if uses_conv()
            else None
        )
        t_prev_add_fc = (
            T.tensor(
                np.array([t.prev_add_fc_obs for t in transitions]), dtype=T.float32
            ).to(self._device)
            if uses_add_fc()
            else None
        )
        t_next_conv = (
            T.tensor(
                np.array([t.next_conv_obs for t in transitions]), dtype=T.float32
            ).to(self._device)
            if uses_conv()
            else None
        )
        t_next_add_fc = (
            T.tensor(
                np.array([t.next_add_fc_obs for t in transitions]), dtype=T.float32
            ).to(self._device)
            if uses_add_fc()
            else None
        )
        t_rewards = T.tensor(
            np.array([t.reward for t in transitions]), dtype=T.float32
        ).to(self._device)

        # creates additional lists that don't require the device.
        next_is_dones = [t.next_is_done for t in transitions]
        action_combo_nums = [action_to_combo_num(t.action) for t in transitions]

        # runs minibatches through the eval neural network to get output.
        minibatch_selector = np.arange(self.PROFILE.MINIBATCH_SIZE, dtype=np.int32)
        t_preds = self._q_eval.forward(t_prev_conv, t_prev_add_fc)[
            minibatch_selector, action_combo_nums
        ]

        # creates the targets tensor.
        # ---
        # if total returns are being used,
        # then the next state doesn't need an approximation.
        if self.PROFILE.USE_TOTAL_RETURNS:
            t_targets = t_rewards

        # if total returns are not being used,
        # then the next states must be approximated.
        else:
            # runs minibatches through the neural networks to get outputs.
            t_evals = self._q_eval.forward(t_next_conv, t_next_add_fc)
            t_nexts = (
                self._q_next.forward(t_next_conv, t_next_add_fc)
                if self._q_next
                else t_evals
            )

            # manually sets the Q-values of illegal actions in the next states
            # to <self.PROFILE.ILLEGAL_VALUE> if specified to do so.
            if self.PROFILE.FORCE_ILLEGALS_IN_NEXTS:
                for i, t in enumerate(transitions):
                    if not next_is_dones[i] and t.prev_state.action_is_legal(t.action):
                        m = t.next_state.create_illegal_subjective_action_mask(
                            perspectives[i]
                        )
                        t_evals[i][m] = -np.Infinity
                        if self._q_next:
                            t_nexts[i][m] = self.PROFILE.ILLEGAL_VALUE

            # sets the Q-values of the terminal states to 0.0.
            t_evals[next_is_dones] = 0.0
            if self._q_next:
                t_nexts[next_is_dones] = 0.0

            # creates a tensor of maximal action selections.
            t_max_actions = (
                T.argmax(t_evals[:, CONST.CONTINUOUS_ACTION_DIM :], dim=1)
                + CONST.CONTINUOUS_ACTION_DIM
            )

            # creates the targets tensor for loss.
            t_targets = (
                t_rewards
                + self.PROFILE.GAMMA * t_nexts[minibatch_selector, t_max_actions]
            )

        # applies specialized selection to the loss tensors if given.
        if self._select_t_preds_method:
            t_preds = self._select_t_preds_method(t_preds)
        if self._select_t_targets_method:
            t_targets = self._select_t_targets_method(t_targets)

        # calculates loss.
        loss = self.PROFILE.LOSS(t_targets, t_preds).to(self._device)

        # calculates regularized loss
        # if specified to do so, then backpropagates.
        if self.PROFILE.L1_REG_STRENGTH > 0.0 and self.PROFILE.L2_REG_STRENGTH > 0.0:
            l1_regularization = T.tensor(0.0, requires_grad=True).to(self._device)
            l2_regularization = T.tensor(0.0, requires_grad=True).to(self._device)
            for param in self._q_eval.parameters():
                l1_regularization += T.norm(param, 1)
                l2_regularization += T.norm(param, 2)

            # calculates total loss.
            reg_loss = (
                loss
                + self._PROFILE.L1_REG_STRENGTH * l1_regularization
                + self._PROFILE.L2_REG_STRENGTH * l2_regularization
            )
            self._loss_history.append(reg_loss.item())
            reg_loss.backward()
        else:
            self._loss_history.append(loss.item())
            loss.backward()

        if self._loss_history[-1] > 100.0:
            print(t_preds)
            print(t_targets)

        # clips gradients.
        if self.PROFILE.GRADIENTS_MAX_NORM > 0.0:
            nn.utils.clip_grad_norm_(
                self._q_eval.parameters(), max_norm=self.PROFILE.GRADIENTS_MAX_NORM
            )

        # steps the optimizer and decrements epsilon.
        self._optimizer.step()

        if self._lr_scheduler:
            self._lr_scheduler.step(self.calc_avg_loss())

        self._decrement_eps_method()
        self._n_learn_steps += 1

        if self.PROFILE.CONTINUOUS_MEMORY:
            self._replace_target_network()
