# panda
## Platform for Agent Network Development and Analysis
This project allows for the quick creation of agent networks (autonomous entities) and testing them in a defined custom environment.

## Workflow
![workflow](https://raw.githubusercontent.com/travisgx/panda/main/_diagrams/_workflow.png)
The idea is that only the CUSTOM files will need modification in order to accomodate any desired game/environment.
<br>
<br>
## State -> Observations
![state to observations](https://raw.githubusercontent.com/travisgx/panda/main/_diagrams/_state_flow.png)
The setup in the source code does not make use of the additional fully-connected observation; it's shown here for demonstrative purposes.
<br>
<br>
## The Neural Process
![neural process](https://github.com/travisgx/panda/blob/main/_diagrams/_layout_panda.png)
panda/constants.py (CONST) and the Player's Profile have settings which influence the neural network's process.
