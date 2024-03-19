'''
panda/environment/action/q_value.py
---
this file contains functions used 
to determine Q-values from network outputs and an action.

'''

import numpy as np
from panda.constants import CONST

# returns the sum of a network's output values 
# selected by <action> within its discrete action spaces.
# this will not be useful if a continuous action space is involved.
def q_value_of_discrete_from_network_output( 
	output, action, illegal_action_mask=None 
):
	discrete_spaces = []
	_append_masked_discrete_action_spaces( 
		discrete_spaces, output, illegal_action_mask
	)
	discrete_q = np.sum( [
		discrete_spaces[action[CONST.CONTINUOUS_ACTION_DIM + i]] 
		for i in range( len( discrete_spaces ))
	] )
	return discrete_q