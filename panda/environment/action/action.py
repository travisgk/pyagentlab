'''
panda/environment/action/action.py
---
this file contains functions used to create action selections
which can be given to a State to execute some transition.

in the context of networks, 'output' will match the specifications
provided by <CONST.CONTINUOUS_ACTION_DIM> and <CONST.DISCRETE_ACTION_DIMS>.

'''

import numpy as np
import torch as T
from panda.constants import CONST

# returns the result of taking an action tuple and converting it
# into a single combination selection number.
def action_to_combo_num( action ):
	discrete_action = action[CONST.CONTINUOUS_ACTION_DIM:]
	combo_num = 0
	product = 1
	for i in range( len( CONST.DISCRETE_ACTION_DIMS ) - 1, -1, -1 ):
		combo_num += discrete_action[i] * product
		product *= CONST.DISCRETE_ACTION_DIMS[i]
	combo_num += CONST.CONTINUOUS_ACTION_DIM
	return combo_num

# returns the result of taking a combination selection number
# and converting it into an action tuple with selections 
# for each distinct, defined discrete action space.
def combo_num_to_action( combo_num ):
	combo_num -= CONST.CONTINUOUS_ACTION_DIM
	discrete_action = [0] * len( CONST.DISCRETE_ACTION_DIMS )
	for i in range( len( CONST.DISCRETE_ACTION_DIMS ) - 1, -1, -1 ):
		discrete_action[i] = combo_num % CONST.DISCRETE_ACTION_DIMS[i]
		combo_num //= CONST.DISCRETE_ACTION_DIMS[i]
	action = ( 0.0, ) * CONST.CONTINUOUS_ACTION_DIM + tuple(discrete_action)
	return action

# returns the action (tuple) composed of the network <outputs> 
# continuous action space outputs and of the indices
# selecting the highest value out of each respective discrete action space.
# <output> should be a one-dimensional numpy array.
# this function is not useful with continuous action spaces.
def subjective_action_from_network_output( output, illegal_action_mask=None ):
	# output_copy = np.array( copy.deepcopy( output ))
	output_copy = np.copy( output )

	# masks illegal actions in the output.
	if illegal_action_mask is not None:
		output_copy[illegal_action_mask] = -np.Infinity
	
	best_discrete_combo_num = np.argmax( 
		output_copy[CONST.CONTINUOUS_ACTION_DIM:] 
	) + CONST.CONTINUOUS_ACTION_DIM

	action = ( tuple(output[:CONST.CONTINUOUS_ACTION_DIM]) + 
			   combo_num_to_action( 
			   	   best_discrete_combo_num 
			   )[CONST.CONTINUOUS_ACTION_DIM:]
	)
	return action

# takes the given network <output>, finds its continuous action space,
# then clips each value within that action space 
# to its respective range specified in CONST.
def _clip_continuous_actions( output ):
	for i in range( CONST.CONTINUOUS_ACTION_DIM ):
		output[i] = np.clip( 
			output[i],
			a_min=CONST.CONTINUOUS_ACTION_LIMITS[i][0], 
			a_max=CONST.CONTINUOUS_ACTION_LIMITS[i][1] 
		)