'''
panda/profile/layer_spec.py
---
this file defines classes which give specifications
of different types of neural network layers.
it also contains functions to create a list of strings
that are used to display information about a list of specifications.

'''

import torch.nn.functional as F

def conv_to_str_list( CONV_LAYER_SPECS ): 
	results = []
	for i, SPEC in enumerate( CONV_LAYER_SPECS ):
		try:
			func_name = SPEC.ACTIVATION_FUNC.__name__
		except AttributeError:
			func_name = SPEC.ACTIVATION_FUNC
		result = (
			f'({SPEC.N_FILTERS:>3d},'
			f' {SPEC.KERNEL_SIZE:>4d},'
			f' {SPEC.STRIDE:>3d}'
			f' {SPEC.PADDING:>3d},'
			f' {str(SPEC.BIAS):>5},'
			f' {SPEC.USE_BATCH_NORM},'
			f' {SPEC.DROPOUT_RATE:>.3f},'
			f' {SPEC.POOLING_SIZE:>3d},'
			f' {func_name})'
		)
		result += ',' if i < len( CONV_LAYER_SPECS ) - 1 else ''
		results.append( result )

	return results

def fc_to_str_list( FC_LAYER_SPECS ):
	results = []
	for i, SPEC in enumerate( FC_LAYER_SPECS ):
		try:
			func_name = SPEC.ACTIVATION_FUNC.__name__
		except AttributeError:
			func_name = SPEC.ACTIVATION_FUNC
		result = (
			f'({SPEC.N_OUT_NODES:>3d},'
			f' {SPEC.USE_BATCH_NORM},'
			f' {SPEC.DROPOUT_RATE:>.3f},'
			f' {func_name})'
		)
		result += ',' if i < len( FC_LAYER_SPECS ) - 1 else ''
		results.append( result )

	return results

class ConvLayerSpec:
	def __init__( 
		self, 
		N_FILTERS, 
		KERNEL_SIZE, 
		STRIDE=1, 
		PADDING=0, 
		BIAS=False,
		USE_BATCH_NORM=False,
		ACTIVATION_FUNC=F.relu,
		DROPOUT_RATE=0.0,
		POOLING_SIZE=0
	):
		self.N_FILTERS = N_FILTERS
		self.KERNEL_SIZE = KERNEL_SIZE
		self.STRIDE = STRIDE
		self.PADDING = PADDING
		self.BIAS = BIAS
		self.USE_BATCH_NORM = USE_BATCH_NORM
		self.ACTIVATION_FUNC = ACTIVATION_FUNC
		self.DROPOUT_RATE = DROPOUT_RATE
		self.POOLING_SIZE = POOLING_SIZE
		
class FClayerSpec:
	def __init__( 
		self, 
		N_OUT_NODES, 
		BIAS=False, 
		USE_BATCH_NORM=False,
		ACTIVATION_FUNC=F.relu, 
		DROPOUT_RATE=0.0, 
	):
		self.N_OUT_NODES = N_OUT_NODES
		self.BIAS = BIAS
		self.USE_BATCH_NORM = USE_BATCH_NORM
		self.ACTIVATION_FUNC = ACTIVATION_FUNC
		self.DROPOUT_RATE = DROPOUT_RATE