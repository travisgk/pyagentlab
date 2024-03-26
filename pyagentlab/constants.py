"""
pyagentlab/constants.py
---
this file defines static variables organized into a single class.
they give specifications for the dimensions of inputs 
and outputs for a network.

these values should be modified externally 
and then <CONST.finalize()> should be run.

"""

import os
import numpy as np


def uses_conv():
    return CONST.CONV_INPUT_DIMS


def uses_add_fc():
    return CONST.ADD_FC_INPUT_DIM


class CONST:
    CHECKPOINT_DIRECTORY = os.path.join("pyagentlab", "models")
    ENV_NAME = "env"

    N_PLAYERS = 1
    CONV_INPUT_DIMS = (3, 3)
    ADD_FC_INPUT_DIM = 0

    # this version of PANDA can facilitate continuous action outputs,
    # however, learning algorithms for continuous actions have not
    # yet been implemented.
    CONTINUOUS_ACTION_DIM = 0
    CONTINUOUS_ACTION_LIMITS = ((-1.0, 1.0),)
    DISCRETE_ACTION_DIMS = (9,)

    # extra variables for easier readability.
    N_CONV_CHANNELS = 0
    CONV_WIDTH = 0
    CONV_HEIGHT = 0
    FLATTENED_DISCRETE_ACTION_DIM = 0

    # if True, playing an illegal move will still change whose turn it is.
    ILLEGAL_MOVE_STEPS_ENV = False

    # environment settings for how match outcomes are determined.
    # ---
    # sets particular player outcomes to end the environment for every player.
    WIN_ENDS_ENTIRE_ENV = False
    LOSS_ENDS_ENTIRE_ENV = False
    ILLEGAL_ENDS_ENTIRE_ENV = False

    # sets other outcome behavior for when a player makes an illegal move.
    ILLEGAL_ENDS_PLAYER = False
    ILLEGAL_END_IS_LOSS = False

    # sets the outcome behavior for the other players when one player's outcome
    # brings the environment to an end for every player.
    # False means that the other players will get an outcome of INTERRUPTED.
    WINS_BY_ENDING_LOSS = True
    WINS_BY_FORFEIT = False
    LOSES_BY_ENDING_WIN = True

    # sets the outcome behavior for when all terminal outcomes match.
    ALL_WINS_IS_DRAW = False
    ALL_LOSSES_IS_DRAW = True

    # logging setting (graphs are not yet implemented).
    ENV_LOG_HISTORY_INTERVAL = 100

    # this function is to be called once CONST has been customized.
    def finalize():
        CONST.N_CONV_CHANNELS = (
            CONST.CONV_INPUT_DIMS[0] if len(CONST.CONV_INPUT_DIMS) >= 1 else 0
        )
        CONST.CONV_WIDTH = (
            CONST.CONV_INPUT_DIMS[1] if len(CONST.CONV_INPUT_DIMS) >= 2 else 0
        )
        CONST.CONV_HEIGHT = (
            CONST.CONV_INPUT_DIMS[2] if len(CONST.CONV_INPUT_DIMS) >= 3 else 0
        )
        CONST.FLATTENED_DISCRETE_ACTION_DIM = np.prod(CONST.DISCRETE_ACTION_DIMS)
        os.makedirs(CONST.CHECKPOINT_DIRECTORY, exist_ok=True)
