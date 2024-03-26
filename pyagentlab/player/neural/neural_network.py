"""
pyagentlab/player/neural/neural_network.py
---
this file defines a class that acts as a neural network
for the NeuralPlayer. the specifications provided by <PROFILE>
define how the neural network's architecture will be built.

"""

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from pyagentlab.constants import CONST, uses_conv, uses_add_fc
from pyagentlab.environment.state import State


class NeuralNetwork(nn.Module):
    def __init__(self, PROFILE, NETWORK_NAME):
        super(NeuralNetwork, self).__init__()
        self._PROFILE = PROFILE
        self.NETWORK_NAME = NETWORK_NAME
        self._CHECKPOINT_FILE = PROFILE.PLAYER_SAVE_PATH + "_" + NETWORK_NAME
        self._create_conv_layers()
        self._create_conv_batch_norms()
        self._create_conv_dropout_layers()

        self._FLATTENS_BEFORE_BC = len(CONST.CONV_INPUT_DIMS) > 1

        self._create_fc_layers()
        self._create_fc_batch_norms()
        self._create_fc_dropout_layers()

    def _create_conv_layers(self):
        self._conv_layers = nn.ModuleList()
        for i, spec in enumerate(self._PROFILE.CONV_LAYER_SPECS):
            if i == 0:
                in_filters = CONST.CONV_INPUT_DIMS[0]
                out_filters = spec.N_FILTERS
            else:
                in_filters = self._PROFILE.CONV_LAYER_SPECS[i - 1].N_FILTERS
                out_filters = spec.N_FILTERS

            self._conv_layers.append(
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    spec.KERNEL_SIZE,
                    stride=spec.STRIDE,
                    padding=spec.PADDING,
                    bias=spec.BIAS,
                )
            )

    def _create_conv_batch_norms(self):
        self._conv_norms = nn.ModuleList(
            [
                nn.BatchNorm2d(conv_layer.state_dict()["weight"].size(0))
                for conv_layer in self._conv_layers
            ]
        )

    def _create_conv_dropout_layers(self):
        self._conv_dropouts = nn.ModuleList(
            [
                nn.Dropout2d(p=spec.DROPOUT_RATE)
                for spec in self._PROFILE.CONV_LAYER_SPECS
            ]
        )

    def _create_fc_layers(self):
        n_conv_output_dims = self._calc_n_conv_outputs()
        N_INPUTS = n_conv_output_dims + CONST.ADD_FC_INPUT_DIM
        N_OUTPUTS = CONST.CONTINUOUS_ACTION_DIM + np.prod(CONST.DISCRETE_ACTION_DIMS)

        self._fc_layers = nn.ModuleList()

        # if there are no node amounts given,
        # then the only layer used directly links the inputs to outputs.
        if len(self._PROFILE.FC_LAYER_SPECS) == 0:
            self._fc_layers.append(nn.Linear(N_INPUTS, N_OUTPUTS, bias=True))

        # otherwise, layers are created as normal.
        else:
            for i, spec in enumerate(self._PROFILE.FC_LAYER_SPECS):
                if i == 0:
                    in_filters = N_INPUTS
                    out_filters = spec.N_OUT_NODES
                else:
                    in_filters = self._PROFILE.FC_LAYER_SPECS[i - 1].N_OUT_NODES
                    out_filters = spec.N_OUT_NODES
                self._fc_layers.append(
                    nn.Linear(in_filters, out_filters, bias=spec.BIAS)
                )

            self._fc_layers.append(
                nn.Linear(self._PROFILE.FC_LAYER_SPECS[-1].N_OUT_NODES, N_OUTPUTS)
            )

    def _calc_n_conv_outputs(self):
        if len(self._conv_layers) == 0:
            # if there are no convolutional layers,
            # then the flattened length is used.
            n_conv_outputs = (
                np.prod(CONST.CONV_INPUT_DIMS)
                if len(CONST.CONV_INPUT_DIMS) > 1
                else CONST.CONV_INPUT_DIMS[0]
            )
        else:
            # if there are any convolutional layers,
            # then a blank observation is passed through
            # and flattened to determine the amount of outputs.
            t_conv_obs = T.tensor(np.expand_dims(State.BLANK_CONV_OBS, axis=0))
            with T.no_grad():
                conv_output = self._conv_layers[0](t_conv_obs)
                for conv_layer in self._conv_layers[1:]:
                    conv_output = conv_layer(conv_layer)
            n_conv_outputs = np.prod(conv_output.size())
        return n_conv_outputs

    def _create_fc_batch_norms(self):
        self._fc_norms = nn.ModuleList(
            [nn.BatchNorm1d(fc_layer.out_features) for fc_layer in self._fc_layers]
        )

    def _create_fc_dropout_layers(self):
        self._fc_dropouts = nn.ModuleList(
            [nn.Dropout(p=spec.DROPOUT_RATE) for spec in self._PROFILE.FC_LAYER_SPECS]
        )

    def save_checkpoint(self, addendum=""):
        print(self._CHECKPOINT_FILE + addendum)
        T.save(self.state_dict(), self._CHECKPOINT_FILE + addendum)

    def load_checkpoint(self, addendum=""):
        print("loading " + self._CHECKPOINT_FILE + addendum)
        self.load_state_dict(T.load(self._CHECKPOINT_FILE + addendum))

    def forward(self, conv_obs, add_fc_obs):
        # 1) convolutional layers.
        if uses_conv():
            result = conv_obs

            for i, conv_layer in enumerate(self._conv_layers):
                SPEC = self._PROFILE.CONV_LAYER_SPECS[i]
                result = conv_layer(result)

                if SPEC.USE_BATCH_NORM and result.shape[0] > 1:
                    result = self._conv_norms[i](result)

                if SPEC.ACTIVATION_FUNC:
                    result = SPEC.ACTIVATION_FUNC(result)

                if SPEC.DROPOUT_RATE > 0.0:
                    result = self._conv_dropouts[i](result)

                if SPEC.POOLING_SIZE > 0 and i < len(self._conv_layers) - 1:
                    result = F.max_pool2d(result, SPEC.POOLING_SIZE)

            # 2) result is flattened.
            if self._FLATTENS_BEFORE_BC:
                result = result.reshape(result.shape[0], -1)

            # 3) concatenates additional FC inputs along the second dimension.
            if uses_add_fc():
                result = T.cat((result, add_fc_obs), dim=1)
        else:
            result = add_fc_obs

        # 4) fully-connected hidden layers.
        for i, fc_layer in enumerate(self._fc_layers[:-1]):
            SPEC = self._PROFILE.FC_LAYER_SPECS[i]
            result = fc_layer(result)

            if SPEC.USE_BATCH_NORM and result.shape[0] > 1:
                result = self._fc_norms[i](result)

            if SPEC.ACTIVATION_FUNC:
                result = SPEC.ACTIVATION_FUNC(result)

            if SPEC.DROPOUT_RATE > 0.0:
                result = self._fc_dropouts[i](result)

        # 5) output layer.
        result = self._fc_layers[-1](result)

        if self._PROFILE.OUTPUT_USE_BATCH_NORM:
            result = self._fc_norms[-1](result)

        if self._PROFILE.OUTPUT_ACTIVATION_FUNC:
            result = self._PROFILE.OUTPUT_ACTIVATION_FUNC(result)

        return result
