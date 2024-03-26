"""
pyagentlab/profile/neural_profile.py
---
this file defines a class that specifies the architecture, 
learning process, and reward mechanism for a neural network agent.

---
-<REPLACE_TARGET_INTERVAL> being 0 will disable the use of a target network.

-<USE_DUELING_ARCHITECTURE> needs at least one specified FC layer to be used.

-<GRADIENTS_MAX_NORM>, <L1_REG_STRENGTH>, and <L2_REG_STRENGTH>
 must be more than 0.0 in order to be used.
 
"""

import copy
import torch.nn as nn
from .profile import Profile
from ._option_categories import _OptionCategories
from .layer_spec import conv_to_str_list, fc_to_str_list


class NeuralProfile(Profile):
    _CATEGORIES = None
    _categorized = False

    def __init__(
        self,
        # outcome settings.
        WIN_VALUE=0.0,
        DRAW_VALUE=0.0,
        LOSS_VALUE=0.0,
        ILLEGAL_VALUE=0.0,
        WIN_VALUE_RANK_FACTOR=1.0,
        LOSS_VALUE_RANK_FACTOR=1.0,
        # epsilon settings.
        EPS_START=0.99,
        EPS_END=0.01,
        EPS_DEC=0.0001,
        # learn settings.
        ENFORCE_LEGALITY=False,
        ENFORCE_LEGALITY_ON_RANDOM=True,
        USE_TOTAL_RETURNS=False,
        GAMMA=0.99,
        MINIBATCH_SIZE=32,
        REPLACE_TARGET_INTERVAL=0,
        FORCE_ILLEGALS_IN_NEXTS=True,
        # learn rate settings.
        LR=0.1,
        LR_SCHEDULER_FACTOR=0.0,
        LR_SCHEDULER_PATIENCE=10000,
        # optimizer settings.
        ALPHA=0.0,
        GRADIENT_MOMENTUM=0.0,
        WEIGHT_DECAY=0.0,
        MIN_SQUARED_GRADIENT=1e-8,
        GRADIENTS_MAX_NORM=-1.0,
        # convolutional layer settings.
        CONV_LAYER_SPECS=[],
        # fully-connected layer settings.
        FC_LAYER_SPECS=[],
        USE_DUELING_ARCHITECTURE=False,
        OUTPUT_USE_BATCH_NORM=False,
        OUTPUT_ACTIVATION_FUNC=None,
        # loss settings.
        LOSS=nn.MSELoss(),
        L1_REG_STRENGTH=-1.0,
        L2_REG_STRENGTH=-1.0,
        # memory settings.
        STARTING_MEM_COUNT=1000,
        MAX_MEM_SIZE=10000,
        CONTINUOUS_MEMORY=False,
        # save directory settings.
        ALGORITHM_NAME="neural",
    ):
        super().__init__(
            WIN_VALUE=WIN_VALUE,
            DRAW_VALUE=DRAW_VALUE,
            LOSS_VALUE=LOSS_VALUE,
            ILLEGAL_VALUE=ILLEGAL_VALUE,
            WIN_VALUE_RANK_FACTOR=WIN_VALUE_RANK_FACTOR,
            LOSS_VALUE_RANK_FACTOR=WIN_VALUE_RANK_FACTOR,
            EPS_START=EPS_START,
            EPS_END=EPS_END,
            EPS_DEC=EPS_DEC,
            ENFORCE_LEGALITY=ENFORCE_LEGALITY,
            ENFORCE_LEGALITY_ON_RANDOM=ENFORCE_LEGALITY_ON_RANDOM,
            USE_TOTAL_RETURNS=USE_TOTAL_RETURNS,
            GAMMA=GAMMA,
            MINIBATCH_SIZE=MINIBATCH_SIZE,
            LR=LR,
            STARTING_MEM_COUNT=STARTING_MEM_COUNT,
            MAX_MEM_SIZE=MAX_MEM_SIZE,
            CONTINUOUS_MEMORY=CONTINUOUS_MEMORY,
            ALGORITHM_NAME=ALGORITHM_NAME,
        )
        if not NeuralProfile._CATEGORIES:
            category_order = [
                "outcome settings",
                "epsilon settings",
                "learn settings",
                "learn rate settings",
                "optimizer settings",
                "convolutional layer settings",
                "fully-connected layer settings",
                "loss settings",
                "memory settings",
                "save directory settings",
            ]
            NeuralProfile._CATEGORIES = copy.deepcopy(Profile._CATEGORIES)
            NeuralProfile._CATEGORIES.rearrange_categories(category_order)

        self.REPLACE_TARGET_INTERVAL = REPLACE_TARGET_INTERVAL
        self.FORCE_ILLEGALS_IN_NEXTS = FORCE_ILLEGALS_IN_NEXTS
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "learn settings"
            )

        self.LR_SCHEDULER_FACTOR = LR_SCHEDULER_FACTOR
        self.LR_SCHEDULER_PATIENCE = LR_SCHEDULER_PATIENCE
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "learn rate settings"
            )

        self.ALPHA = ALPHA
        self.GRADIENT_MOMENTUM = GRADIENT_MOMENTUM
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.MIN_SQUARED_GRADIENT = MIN_SQUARED_GRADIENT
        self.GRADIENTS_MAX_NORM = GRADIENTS_MAX_NORM
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "optimizer settings"
            )

        self.CONV_LAYER_SPECS = CONV_LAYER_SPECS
        self.CONV_LAYERS_STR_LIST = conv_to_str_list(self.CONV_LAYER_SPECS)
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "convolutional layer settings"
            )

        self.FC_LAYER_SPECS = FC_LAYER_SPECS
        self.USE_DUELING_ARCHITECTURE = USE_DUELING_ARCHITECTURE
        self.OUTPUT_USE_BATCH_NORM = OUTPUT_USE_BATCH_NORM
        self.OUTPUT_ACTIVATION_FUNC = OUTPUT_ACTIVATION_FUNC
        self.FC_LAYERS_STR_LIST = fc_to_str_list(self.FC_LAYER_SPECS)
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "fully-connected layer settings"
            )

        self.LOSS = LOSS
        self.L1_REG_STRENGTH = L1_REG_STRENGTH
        self.L2_REG_STRENGTH = L2_REG_STRENGTH
        if not NeuralProfile._categorized:
            NeuralProfile._CATEGORIES.organize_new_keys(
                self.__dict__.keys(), "loss settings"
            )

        NeuralProfile._categorized = True

    # appends (to the given list) lists of strings
    # displaying the Player's settings.
    def append_category_lists_to(self, info_lists):
        NeuralProfile._CATEGORIES.append_category_lists_to(info_lists, self.__dict__)
