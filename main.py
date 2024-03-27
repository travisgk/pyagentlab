from collections import deque
import numpy as np
import torch as T
import torch.nn as nn
from pyagentlab import *
from gomoku_state import GomokuState


# runs many episodes of Tic-Tac-Toe with one neural network playing itself.
# first, <Const> is set to the desired specifications.
# then, the profiles are set up and players are created.
# next, the environment is created,
# and finally, the episodes are run.
def main():
    Const.ENV_NAME = "Gomoku"
    width, height = 3, 3
    Const.N_PLAYERS = 2
    Const.CONV_INPUT_DIMS = (Const.N_PLAYERS + 1, width, height)
    Const.ADD_FC_INPUT_DIM = 0
    Const.DISCRETE_ACTION_DIMS = (width, height)
    Const.CONTINUOUS_ACTION_DIM = 0
    Const.WIN_ENDS_ENTIRE_ENV = True
    Const.finalize()
    GomokuState.WIN_LENGTH = 3

    NEURAL_PROFILE = NeuralProfile(
        # outcome settings.
        WIN_VALUE=1.0,
        DRAW_VALUE=1.0,
        LOSS_VALUE=0.0,
        ILLEGAL_VALUE=-0.1,
        WIN_VALUE_RANK_FACTOR=0.9,
        LOSS_VALUE_RANK_FACTOR=0.9,
        # epsilon settings.
        EPS_START=0.7,
        EPS_DEC=5e-6,
        EPS_END=0.05,
        # learn settings.
        ENFORCE_LEGALITY=False,
        ENFORCE_LEGALITY_ON_RANDOM=True,
        USE_TOTAL_RETURNS=False,
        GAMMA=0.8,
        MINIBATCH_SIZE=6,
        REPLACE_TARGET_INTERVAL=0,
        FORCE_ILLEGALS_IN_NEXTS=True,
        # learn rate settings.
        LR=0.001,
        LR_SCHEDULER_FACTOR=0.5,
        LR_SCHEDULER_PATIENCE=50000,
        # optimizer settings.
        ALPHA=0.0,
        GRADIENT_MOMENTUM=0.0,
        WEIGHT_DECAY=0.0,
        MIN_SQUARED_GRADIENT=1e-8,
        GRADIENTS_MAX_NORM=-1.0,
        # convolutional layer settings.
        CONV_LAYER_SPECS=[],
        # fully-connected layer settings.
        FC_LAYER_SPECS=[FClayerSpec(128, BIAS=True)],
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
    )
    neural_player = NeuralPlayer(NEURAL_PROFILE)

    RANDOM_PROFILE = Profile(ENFORCE_LEGALITY=True)
    random_player = Player(RANDOM_PROFILE)

    players = [neural_player, neural_player]

    env = Environment(StateClass=GomokuState)

    print("Beginning to run episodes.")
    play_episodes(100000, env, players, is_training=True)
    neural_player.save_checkpoints()


if __name__ == "__main__":
    main()
