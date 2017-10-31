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
		try:
			dim == int(dim)
		except ValueError:
			pass  # not an int.
		self.dim = dim
		self.alpha = alpha
		self.p = self.data.shape[1] #elements
		self.n = self.data.shape[0] #variables

	def standardize(self):
		medias = self.data.mean(axis=0)
		desv = self.data.std(axis=0)
		self.data_st = (self.data-medias)/desv
		return self.data_st

	#def SVD(self.data):

