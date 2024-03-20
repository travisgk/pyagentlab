# panda (version 0.1)
## Platform for Agent Network Development and Analysis
This project allows for the quick creation of agent networks (autonomous entities) and testing them in a defined custom environment.

## Workflow
![workflow](https://raw.githubusercontent.com/travisgx/panda/main/_diagrams/_workflow.png)
The idea is that only the CUSTOM files will need modification in order to accomodate any desired game/environment.
<br>
<br>
## Gomoku
The example of a custom State in this source code implements the game of Gomoku. This is a game where both players try to be the first to place some amount of their own markers in a row. 

Mathematically, this is known as an "m,n,k-game", where players each try to make a line of k-in-a-row on a board of size m by n. Gomoku is a 15,15,5-game, and Tic-Tac-Toe is a 3,3,3-game. "Gomoku" is used as a simple catch-all term in this case. All of these values, including the amount of players, can be changed from the main.py file.
<br>
<br>
## State -> Observations
![state to observations](https://raw.githubusercontent.com/travisgx/panda/main/_diagrams/_state_flow.png)
The setup in the source code does not make use of the additional fully-connected observation; it's shown here for demonstrative purposes.
<br>
<br>
## The Neural Process
![neural process](https://raw.githubusercontent.com/travisgx/panda/main/_diagrams/_layout_panda.png)
panda/constants.py (CONST) and the Player's Profile have settings which influence the neural network's process. 

This system is set up to accommodate a continuous action space, however, there is not yet any implementation for the neural network to learn the best continuous actions. As such, the step of choosing the best action here only finds the optimal discrete action selections.
