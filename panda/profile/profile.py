'''
panda/profile/profile.py
---
this file defines a class that specifies the learning process
and reward mechanism for a generic agent.

---
-<WIN_VALUE> will be multiplied with the exponential <WIN_VALUE_RANK_FACTOR>
 for each rank below 1 in order to calculate the Player's reward for winning.

-<LOSS_VALUE> will be multiplied with the exponential <LOSS_VALUE_RANK_FACTOR>
 for each rank above the last rank in order 
 to calculate the Player's reward for winning.
 
'''

import os
from panda.constants import CONST
from panda.profile._option_categories import _OptionCategories

class Profile:
	_CATEGORIES = _OptionCategories()
	_categorized = False
	def __init__( 
		self,

		# outcome settings.
		WIN_VALUE=0.0,
		DRAW_VALUE=0.0,
		LOSS_VALUE=0.0,
		ILLEGAL_VALUE=0.0,
		WIN_VALUE_RANK_FACTOR=1.0,
		LOSS_VALUE_RANK_FACTOR=1.0,

		# epsilon settings.
		EPS_START=0.99,
		EPS_END=0.01,
		EPS_DEC=0.0001,

		# learn settings.
		ENFORCE_LEGALITY=False,
		ENFORCE_LEGALITY_ON_RANDOM=True,
		USE_TOTAL_RETURNS=False,
		GAMMA=0.99,
		MINIBATCH_SIZE=32,

		# learn rate settings.
		LR=0.1,

		# memory settings.
		STARTING_MEM_COUNT=1000,
		MAX_MEM_SIZE=10000,
		CONTINUOUS_MEMORY=False,

		# save directory settings.
		ALGORITHM_NAME='random'
	):
		self.WIN_VALUE = WIN_VALUE
		self.DRAW_VALUE = DRAW_VALUE
		self.LOSS_VALUE = LOSS_VALUE
		self.ILLEGAL_VALUE = ILLEGAL_VALUE
		self.WIN_VALUE_RANK_FACTOR = WIN_VALUE_RANK_FACTOR
		self.LOSS_VALUE_RANK_FACTOR = LOSS_VALUE_RANK_FACTOR
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'outcome settings'
			)

		self.EPS_START = EPS_START
		self.EPS_END = EPS_END
		self.EPS_DEC = EPS_DEC
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'epsilon settings'
			)

		self.ENFORCE_LEGALITY = ENFORCE_LEGALITY
		self.ENFORCE_LEGALITY_ON_RANDOM = ENFORCE_LEGALITY_ON_RANDOM
		self.USE_TOTAL_RETURNS = USE_TOTAL_RETURNS
		self.GAMMA = GAMMA
		
		self.MINIBATCH_SIZE = MINIBATCH_SIZE
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'learn settings'
			)

		self.LR = LR
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'learn rate settings'
			)

		self.STARTING_MEM_COUNT = STARTING_MEM_COUNT
		self.MAX_MEM_SIZE = MAX_MEM_SIZE
		self.CONTINUOUS_MEMORY = CONTINUOUS_MEMORY
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'memory settings'
			)
		
		self.ALGORITHM_NAME = ALGORITHM_NAME
		self.PLAYER_SAVE_PATH = os.path.join( 
			CONST.CHECKPOINT_DIRECTORY, 
			f'{CONST.ENV_NAME}_{self.ALGORITHM_NAME}'
		)
		if not Profile._categorized:
			Profile._CATEGORIES.organize_new_keys( 
				self.__dict__.keys(), 'save directory settings'
			)
		Profile._categorized = True

	# appends (to the given list) lists of strings 
	# displaying the Player's settings.
	def append_category_lists_to( self, info_lists ):
		Profile._CATEGORIES.append_category_lists_to( 
			info_lists, self.__dict__
		)