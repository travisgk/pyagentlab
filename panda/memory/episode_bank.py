'''
panda/memory/episode_bank.py
---
this file defines a memory bank that stores Transitions
with the use of Episodes.

'''

from collections import deque
import numpy as np
from panda.memory._transition_bank import TransitionBank

class EpisodeBank( TransitionBank ):
	def __init__( self, PROFILE ):
		super().__init__()
		self._episodes = deque( maxlen=PROFILE.MAX_MEM_SIZE )

	# appends to the given <transitions_list> and <perspectives_list>
	# with <minibatch_size> Transitions and perspectives.
	def sample_transitions( self,
							transitions_list,
							perspectives_list,
							minibatch_size,
							sample_entire_episodes=False
	):
		episode_indices = np.random.choice(
			len( self._episodes ), minibatch_size, replace=False
		)
		
		sample_method = ( self._mass_sample if sample_entire_episodes
						  else self._single_sample
		)

		minibatch_remaining = minibatch_size
		for ep_index in episode_indices:
			sample_method( transitions_list,
						   perspectives_list,
						   ep_index,
						   minibatch_remaining
			)
			minibatch_remaining -= len( self._episodes[ep_index].transitions )
			if sample_entire_episodes and minibatch_remaining <= 0:
				break

	def store_finalized_episode( self, episode ):
		self._episodes.append( episode )

	def _mass_sample( 
		self, transitions_list, perspectives_list, ep_index, max_memories
	):
		episode = self._episodes[ep_index]
		episode_length = len( episode.transitions )
		end = min( max_memories, episode_length )
		transitions_list.extend( [mem for mem in episode.transitions[:end]] )
		perspectives_list.extend(
			np.full( end, episode.perspective, dtype=np.uint8 )
		)

	def _single_sample(
		self, transitions_list, perspectives_list, ep_index, max_memories
	):
		episode = self._episodes[ep_index]
		episode_length = len( episode.transitions )
		index = np.random.choice( episode_length )
		transitions_list.append( episode.transitions[index] )
		perspectives_list.append( episode.perspective )

	def length( self ):
		return len( self._episodes )