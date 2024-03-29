# pyagentlab (v0.1)
This package allows for the quick creation of agent networks (autonomous entities) and testing them in a defined custom environment.
<br>
<br>
## Installation
pyagentlab makes use of [numpy](https://github.com/numpy/numpy) and [pytorch](https://github.com/pytorch/pytorch).
The necessary packages can be installed with pip using the command `pip install numpy torch`.
<br>
<br>
## Usage
### Creating a custom State class
The State class contains methods for an agent network to interface with it, which can be overridden in a child class.
When creating a custom State class, as exemplified in _gomoku_state.py_, these methods can be overridden in order to define rules of the environment, as well as how the custom State gives observations to the agent network.
These methods designed for overriding are:
- `setup()` - initializes static class variables.
- `reset()` - resets member variables at the beginning of an episode.
- `to_conv_obs(perspective)` - returns a multidimensional list that represents the State's convolutional information as viewed from the player #`perspective`. `perspective` may not need to be used inside this method. 
- `to_add_fc_obs(perspective)` - returns a one-dimensional list that represents any additional State information (vital to learning) as viewed from the player #`perspective`. `perspective` may not need to be used inside this method.
- `create_legal_subjective_action_mask()` - returns a one-dimensional list of boolean values that can be used to mask the agent network's discrete action space output values.
- `action_is_legal(action, player_num)` - returns `True` if the given discrete action selection `action` is allowed to be taken by player #`player_num` given the current State. `player_num` may not need to be used inside this method.
- `legal_plays_remain(player_num)` - returns `True` if any legal plays remain, `player_num` can be used inside this method to define player-specific legality.
- `make_action_objective(self, subjective_action, player_num)` - returns a discrete action selection so that the perspective of player #`player_num` is applied. by default, this method will not apply any sort of perspective, simply returning the `subjective_action` selection as is.
- `_apply_action(action, player_num)` - applies a discrete action selection to the State for player #`player_num`.
- `_win_condition(action, player_num, episode_n_steps)` - returns `True` if player #`player_num` reaches a winning position by taking discrete action selection `action` at the episode's `episode_n_steps`-th steps. by default this willl return `False`.
- `_loss_condition(action, player_num, episode_n_steps)` - returns `True` if player #`player_num` reaches a losing position by taking discrete action selection `action` at the episode's `episode_n_steps`-th steps. by default this willl return `False`.
- `_draw_condition(action, player_num, episode_n_steps)` - returns `True` if player #`player_num` reaches a draw position by taking discrete action selection `action` at the episode's `episode_n_steps`-th steps. by default this willl return `False`.
<br>

### Running episodes with a neural network
The module `constants` contains the `Const` class, which holds system configurations that should be set up before creating the environment and agent. For Tic-Tac-Toe, these look like:
```
from pyagentlab import *
from gomoku_state import GomokuState
WIDTH, HEIGHT = 3, 3
Const.ENV_NAME = "Gomoku"
Const.N_PLAYERS = 2

# the dimensions of the inputs given to every agent network are specified.
Const.CONV_INPUT_DIMS = (Const.N_PLAYERS + 1, WIDTH, HEIGHT)
Const.ADD_FC_INPUT_DIM = 0

# the dimensions of the outputs from every agent network are specified.
Const.DISCRETE_ACTION_DIMS = (WIDTH, HEIGHT)
Const.CONTINUOUS_ACTION_DIM = 0

# the behaviors of how a single player reaching
# their own game outcome affects all the other players is set.
Const.WIN_ENDS_ENTIRE_ENV = True

# the Const class has some assisting values computed.
Const.finalize()

# settings for the game of Gomoku are specified.
GomokuState.WIN_LENGTH = 3
```
<br>

The system has now been set up to accomodate the game's custom State.
Now it's time to initialize the environment with the custom State. 
**This must be done before creating the players.**
```
env = Environment(StateClass=GomokuState)
```
<br>

The architecture and behavior of a neural network is specified with a `NeuralProfile` and its contained list of `ConvLayerSpec` objects and `FClayerSpec` objects, then an agent network is created as a `NeuralPlayer`:
```
NEURAL_PROFILE = NeuralProfile(
	WIN_REWARD=1.0,
	LOSS_REWARD=0.0,
	CONV_LAYER_SPECS=[ConvLayerSpec(128, 3)],
	FC_LAYER_SPECS=[FclayerSpec(16, BIAS=True)],
)
neural_player = NeuralPlayer(NEURAL_PLAYER)
```
<br>

An agent network that makes purely random moves can be created with:
```
RANDOM_PROFILE = Profile(ENFORCE_LEGALITY=True)
random_player = Player(RANDOM_PROFILE)
```
<br>

Finally, a list of players is created, the environment is created for the custom State class, and episodes are played:
```
players = [neural_player, random_player]
play_episodes(10000, env, players, is_training=True)

# saves the neural network's weights to file.
neural_player.save_checkpoints();
```
