'''
panda/environment/action/random_action.py
---
this file contains functions used to create random action selections
which can be given to a State to execute some transition.

'''

import numpy as np
from panda.constants import CONST
from panda.environment.action.action import (
	subjective_action_from_network_output
)

# returns an action with random values that fall within CONST specifications.
def random_subjective_action( illegal_action_mask=None ):
	continuous_output = [
		np.random.uniform( CONST.CONTINUOUS_ACTION_LIMITS[i][0], 
						   CONST.CONTINUOUS_ACTION_LIMITS[i][1] 
		) if i < len( CONST.CONTINUOUS_ACTION_LIMITS ) else np.random.rand()
		for i in range( CONST.CONTINUOUS_ACTION_DIM )
	]

	discrete_output = [
		np.random.rand() for _ in range( CONST.FLATTENED_DISCRETE_ACTION_DIM  )
	]

	output = continuous_output + discrete_output

	action = subjective_action_from_network_output(
		output, illegal_action_mask
	)

	return action