'''
panda/profile/_option_categories.py
---
this file, exclusive to the 'profile' module,
defines classes that are used to create static objects
which help organize member variables into distinct categories.
it's used in the Profile class and its children to make outputting
its properties clean and easy to read. 

'''

class _OptionCategory:
	def __init__( self, category_name ):
		self.key_names = []
		self.category_name = category_name

class _OptionCategories:
	def __init__( self ):
		self._to_index = {}
		self._n_categories = 0
		self._categories = []

	def organize_new_keys( self, keys, category_name ):
		category_index = self.add_category( category_name )
		if category_index < 0:
			return
		for key_name in keys:
			if not self._key_is_categorized( key_name ):
				self._categories[category_index].key_names.append( key_name )

	# returns the index of the new category
	# or the index of the existing category.
	def add_category( self, category_name ):
		if ( category_name not 
			 in [category.category_name for category in self._categories]
		):
			self._categories.append( _OptionCategory( category_name ))
			self._to_index[category_name] = self._n_categories
			self._n_categories += 1
			return self._n_categories - 1
		return self._to_index.get( category_name, -1 )

	def _key_is_categorized( self, key_name ):
		for category in self._categories:
			if key_name in category.key_names:
				return True
		return False
	
	# appends (to the given list) lists of strings 
	# which display each category's keys and values.
	# the first string of each sublist will be the category name.
	def append_category_lists_to( self, info_lists, self_dict ):
		TAB = 30
		HIDDEN_KEYS = [
			'__init__', 
			'get_category_lists', 
			'CONV_LAYERS_STR_LIST', 
			'FC_LAYERS_STR_LIST'
		]
		FUNC_KEYS = ['OUTPUT_ACTIVATION_FUNC', 'LOSS']
		keys = self_dict.keys()
		
		categories = [c for c in self._categories if len( c.key_names ) > 0]
		for category in categories:
			info_list = [category.category_name]
			for key_name in category.key_names:
				if key_name not in HIDDEN_KEYS:
					# changes search key if some other variable 
					# is used for string representation.
					if key_name == 'CONV_LAYER_SPECS':
						search_key = 'CONV_LAYERS_STR_LIST'
					elif key_name == 'FC_LAYER_SPECS':
						search_key = 'FC_LAYERS_STR_LIST'
					else:
						search_key = key_name

					# finds the index of <search_key> and appends its info.
					try:
						VALUE = self_dict[search_key]
						
						# appends lines whose keys use scientific notation.
						if search_key in ['LR', 'EPS_DEC']:
							info_list.append( key_name.ljust( TAB ) + 
											  f' {VALUE:.1e}\n'
							)

						# appends lines of layer specifications.
						elif ( search_key in 
							   ['CONV_LAYERS_STR_LIST', 'FC_LAYERS_STR_LIST']
						):
							STR_LIST = VALUE
							if len( STR_LIST ) == 0:
								info_list.append( key_name.ljust( TAB ) + 
												  f' []\n'
								)
							elif len( STR_LIST ) == 1:
								info_list.append( key_name.ljust( TAB ) + 
												  f' [{STR_LIST[0]}]\n'
								)
							else:
								info_list.append( key_name.ljust( TAB ) + 
												  f' [{STR_LIST[0]}'
								)
								for line in STR_LIST[1:]:
									info_list.append( 
										' ' * TAB + f'  {line}\n' 
									)
								info_list.append( ' ' * TAB + ' ]\n' )

						# appends the name of a function.
						elif search_key in FUNC_KEYS:
							try:
								func_name = VALUE.__name__
							except AttributeError:
								func_name = VALUE
							info_list.append( key_name.ljust( TAB ) + 
											  f' {func_name}\n' 
							)

						# appends line of information as usual.
						else:
							info_list.append( key_name.ljust( TAB ) + 
											  f' {VALUE}\n' 
							)
					except ValueError:
						pass
			info_lists.append( info_list )

	# changes the order of the categories by providing a new order.
	def rearrange_categories( self, category_names ):
		# adds new categories that are not yet in the list.
		for category_name in category_names:
			self.add_category( category_name )

		# sorts all the categories in the given order of <category_names>.
		self._categories = sorted( 
			self._categories, 
			key=lambda x: category_names.index( x.category_name ) 
				if x.category_name in category_names else len( category_names )
		)
