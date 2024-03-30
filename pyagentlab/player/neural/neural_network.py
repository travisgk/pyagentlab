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
from pyagentlab.constants import Const, uses_conv, uses_add_fc
from pyagentlab.environment.state import State


class NeuralNetwork(nn.Module):
    # init. <PROFILE> is used to build the neural network's achitecture.
    def __init__(self, PROFILE, NETWORK_NAME):
        super(NeuralNetwork, self).__init__()
        self._PROFILE = PROFILE
        self._NETWORK_NAME = NETWORK_NAME
        self._CHECKPOINT_FILE = PROFILE.PLAYER_SAVE_PATH + "_" + NETWORK_NAME
        self._create_conv_layers()
        self._create_conv_batch_norms()
        self._create_conv_dropout_layers()
        self._create_fc_layers()
        self._create_fc_batch_norms()
        self._create_fc_dropout_layers()

    def _create_conv_layers(self):
        self._conv_layers = nn.ModuleList()
        for i, spec in enumerate(self._PROFILE.CONV_LAYER_SPECS):
            if i == 0:
                in_filters = Const.CONV_INPUT_DIMS[0]
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

    # creates the fully-connected layers of the neural network.
    # if <self._PROFILE> has no specifications for the fully-connected layers,
    # then a single layer connecting the number of input nodes
    # to the number of output nodes is created.
    #
    # the final fully-connected layer doesn't need to be specified
    # by the <self._PROFILE>, the method will do this automatically
    # to match the number of output nodes.
    #
    # the number of input nodes is:
    #   the dimension of the flattened convolutional observation
    #   + the dimension of the additional fully-connected observation.
    #
    # and the number of output nodes is:
    #   the dimension of the continuous actions
    #   + the dimension of the flattened discrete action spaces.
    def _create_fc_layers(self):
        flat_conv_output_dim = self._calc_flattened_conv_output_dimension()
        N_INPUTS = flat_conv_output_dim + Const.ADD_FC_INPUT_DIM
        N_OUTPUTS = Const.CONTINUOUS_ACTION_DIM + Const.FLATTENED_DISCRETE_ACTION_DIM
        self._fc_layers = nn.ModuleList()

        if len(self._PROFILE.FC_LAYER_SPECS) == 0:
            self._fc_layers.append(nn.Linear(N_INPUTS, N_OUTPUTS, bias=True))
            return

        # builds specified fully-connected layers.
        for i, spec in enumerate(self._PROFILE.FC_LAYER_SPECS):
            if i == 0:
                in_filters = N_INPUTS
                out_filters = spec.N_OUT_NODES
            else:
                in_filters = self._PROFILE.FC_LAYER_SPECS[i - 1].N_OUT_NODES
                out_filters = spec.N_OUT_NODES
            self._fc_layers.append(nn.Linear(in_filters, out_filters, bias=spec.BIAS))

        # final layer.
        self._fc_layers.append(
            nn.Linear(self._PROFILE.FC_LAYER_SPECS[-1].N_OUT_NODES, N_OUTPUTS)
        )

    # returns the single dimension of the flattened convolutional output.
    # ---
    # if there are no convolutional layers, then the dimension of the
    # convolutional output can be easily determined by flattening it.
    #
    # otherwise, a blank convolutional observation is passed through
    # the convolutional layers, and the flattened dimension of that output
    # is returned.
    def _calc_flattened_conv_output_dimension(self):
        if len(self._conv_layers) == 0:
            flattened_conv_output_dim = np.prod(Const.CONV_INPUT_DIMS)
            return flattened_conv_output_dim

        t_conv_obs = T.tensor(np.expand_dims(State.BLANK_CONV_OBS, axis=0))
        with T.no_grad():
            conv_output = self._conv_layers[0](t_conv_obs)
            for conv_layer in self._conv_layers[1:]:
                conv_output = conv_layer(conv_output)

        flattened_conv_output_dim = np.prod(conv_output.size())
        return flattened_conv_output_dim

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

    # returns the output of the final layer of the neural network
    # after performing forward propagation
    # with the given convolutional observation <conv_obs>
    # and the additional fully-connected observation <add_fc_obs>.
    # ---
    # the <conv_obs> is first passed through the convolutional layers,
    # then the convolutional layers output is flattened.
    # next, the <add_fc_obs> is concatenated to the flattened output.
    # finally, the output is fed into the fully-connected layers
    # and the output of the final layer is returned.
    def forward(self, conv_obs, add_fc_obs):
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

            # the convolutional output is flattened.
            if len(Const.CONV_INPUT_DIMS) > 1:
                result = result.reshape(result.shape[0], -1)

            if uses_add_fc():
                result = T.cat((result, add_fc_obs), dim=1)
        else:
            result = add_fc_obs

        # the output is run through the fully-connected layers.
        for i, fc_layer in enumerate(self._fc_layers[:-1]):
            SPEC = self._PROFILE.FC_LAYER_SPECS[i]
            result = fc_layer(result)

            if SPEC.USE_BATCH_NORM and result.shape[0] > 1:
                result = self._fc_norms[i](result)

            if SPEC.ACTIVATION_FUNC:
                result = SPEC.ACTIVATION_FUNC(result)

            if SPEC.DROPOUT_RATE > 0.0:
                result = self._fc_dropouts[i](result)

        # the output of the final fully-connected layer is returned.
        result = self._fc_layers[-1](result)

        if self._PROFILE.OUTPUT_USE_BATCH_NORM:
            result = self._fc_norms[-1](result)

        if self._PROFILE.OUTPUT_ACTIVATION_FUNC:
            result = self._PROFILE.OUTPUT_ACTIVATION_FUNC(result)

        return result
