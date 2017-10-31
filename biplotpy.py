#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

class biplotpy:
	'''
	Gabriel Biplots
	'''

	def __init__(self, data,dim,alpha = 1):
		self.data = data
		if isinstance(dim, (int, long, float)):
			self.dim = dim
		else:
			raise Exception('not numeric')
		if (alpha>=0 and alpha<=1):
			self.alpha = alpha
		else:
			raise Exception('not between 0 and 1')
		self.p = self.data.shape[1] #elements
		self.n = self.data.shape[0] #variables

	def standardize(self):
		medias = self.data.mean(axis=0)
		desv = self.data.std(axis=0)
		self.data_st = (self.data-medias)/desv
		return self.data_st

	#def SVD(self.data):

