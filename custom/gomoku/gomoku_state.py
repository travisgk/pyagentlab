'''
panda/environment/gomoku_state.py
---
this file defines a custom State in the environment.
<self.conv> will contain one-hot encoded information 
to represent a Gomoku board. any format can be used
to hold 2D information, so long as it's later
transformed into some observation a network can read.

this example uses one-hot encoding by default
to cut down on computations.

'''

import copy
import numpy as np
from panda.constants import CONST, uses_conv, uses_add_fc
from panda.environment.outcome import OUTCOME
from panda.environment.state import State, _apply_perspective

class GomokuState( State ):
	LINES = None
	LINES_BY_CELL = None
	WIN_LENGTH = 3
	MARKERS = ['.', 'X', 'O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	def __init__( self ):
		super().__init__()

	def __copy__( self ):
		copy_state = GomokuState()
		
		copy_state.conv = copy.deepcopy( self.conv ) if uses_conv() else None
		copy_state.add_fc = copy.deepcopy( self.add_fc ) if uses_add_fc() else None
		
		#copy_state.conv = np.copy( self.conv ) if uses_conv() else None
		#copy_state.add_fc = np.copy( self.add_fc ) if uses_add_fc() else None
		return copy_state

	# sets up static aspects of this State class.
	def setup():
		State.setup()
		GomokuState._setup_lines()

	# sets up 
	def reset( self ):
		super().reset()
		self.conv = np.copy( State.BLANK_CONV_OBS ) if uses_conv() else None
		self.add_fc = ( np.copy( State.BLANK_ADD_FC_OBS ) 
						if uses_add_fc() else None
		)

		for x in range( CONST.CONV_WIDTH ):
			for y in range( CONST.CONV_HEIGHT ):
				self.conv[0][x][y] = State.ONE_HOT_TRUE

	def to_conv_obs( self, perspective ):
		return [
			self.conv[_apply_perspective( i, perspective )]
			for i in range( 1 + CONST.N_PLAYERS )
		]

	def to_add_fc_obs( self, perspective ):
		return self.add_fc

	def create_legal_subjective_action_mask( self, player_num ):
		mask = np.zeros( CONST.CONTINUOUS_ACTION_DIM + CONST.FLATTENED_DISCRETE_ACTION_DIM, dtype=bool )
		flattened = np.ravel( self.conv[0] )
		mask[CONST.CONTINUOUS_ACTION_DIM:] = ( flattened == State.ONE_HOT_TRUE )
		
		return mask

	def action_is_legal( self, action, player_num=None ):
		x, y = action[-2:]
		return self.conv[0][x][y] == State.ONE_HOT_TRUE

	def legal_plays_remain( self, player_num=None ):
		return self._has_open_spaces()

	def _apply_action( self, action, player_num=None ):
		x, y = action[-2:]
		self.conv[0][x][y] = State.ONE_HOT_FALSE
		self.conv[player_num][x][y] = State.ONE_HOT_TRUE

	def _win_condition( self, action, player_num=None, local_n_steps=None ):
		x, y = action[-2:]
		return (
			local_n_steps >= GomokuState.WIN_LENGTH - 1 and
			self._has_line( player_num, GomokuState.WIN_LENGTH, ( x, y ))
		)

	def to_str( self ):
		LEFT_SPACING = 9
		result = ''
		for y in range( CONST.CONV_HEIGHT ):
			result += ' ' * LEFT_SPACING
			for x in range( CONST.CONV_WIDTH ):
				char = GomokuState.MARKERS[0]
				for l in range( 1, 1 + CONST.N_PLAYERS ):
					if self.conv[l][x][y] == State.ONE_HOT_TRUE:
						char = GomokuState.MARKERS[l]
						break
				result += f'{char} '
			result += '\n'
		return result

	# specialized methods to GomokuState.
	# ---
	# returns True if there are any open spaces left.
	def _has_open_spaces( self ):
		return any( self.conv[0][x][y] == State.ONE_HOT_TRUE
					for x in range( CONST.CONV_WIDTH )
					for y in range( CONST.CONV_HEIGHT )
		)

	# returns True if there's a line of at least <line_length>
	# composed by <player_num> after taking <last_action_xy>.
	def _has_line( self, player_num, line_length, last_action_xy ):
		l_x, l_y = last_action_xy
		for line in GomokuState.LINES_BY_CELL[l_x][l_y]:
			found_length = 0
			for x, y in line:
				if self.conv[player_num][x][y] == State.ONE_HOT_FALSE:
					found_length = 0
				else:
					found_length += 1
					if found_length >= line_length:
						return True
		return False

	# initializes static lists to quickly determine if a line has been formed.
	def _setup_lines():
		W, H = CONST.CONV_WIDTH, CONST.CONV_HEIGHT
		GomokuState.LINES = []

		# creates rows.
		GomokuState.LINES.extend(
			[[( x, y ) for y in range( H )] for x in range( W )]
		)

		# creates columns.
		GomokuState.LINES.extend(
			[[( x, y ) for x in range( W )] for y in range( H )]
		)

		# creates falling diagonals.
		top_left_cells = (
			[( x, 0 ) for x in range( W - GomokuState.WIN_LENGTH + 1 )] +
			[( 0, y ) for y in range( 1, H - GomokuState.WIN_LENGTH + 1 )]
		)
		for start in top_left_cells:
			GomokuState.LINES.append( [] )
			x = start[0]
			y = start[1]
			while x < W and y < H:
				GomokuState.LINES[-1].append(( x, y ))
				x += 1
				y += 1

		# creates rising diagonals.
		bottom_left_cells = (
			[( x, H - 1 ) for x in range( W - GomokuState.WIN_LENGTH + 1 )] +
			[( 0, y ) for y in range( 1, H - GomokuState.WIN_LENGTH + 1 )]
		)
		for start in bottom_left_cells:
			GomokuState.LINES.append( [] )
			x = start[0]
			y = start[1]
			while x < W and y >= 0:
				GomokuState.LINES[-1].append(( x, y ))
				x += 1
				y -= 1

		# defines <GomokuState.LINES_BY_CELL>.
		GomokuState.LINES_BY_CELL = [
			[[] for y in range( H )] for x in range( W )
		]
		for x in range( W ):
			for y in range( H ):
				for line in GomokuState.LINES:
					if ( x, y ) in line:
						GomokuState.LINES_BY_CELL[x][y].append( line )