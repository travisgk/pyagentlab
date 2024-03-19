'''
panda/memory/_transition_bank.py
---
this file defines a generic bank for holding Transitions
and a method to sample those transitions randomly.
it's intended to be used as an abstract parent class.

'''

class TransitionBank:
	def __init__( self ):
		return

	# appends to the given <transitions_list> and <perspectives_list>
	# with <minibatch_size> Transitions and perspectives.
	def sample_transitions( self,
							transitions_list,
							perspectives_list,
							minibatch_size,
							sample_entire_episodes=False
	):
		return

	def length( self ):
		return 0