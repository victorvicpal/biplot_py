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
		self.p = self.data.shape[1] #elements
		self.n = self.data.shape[0] #variables
		if isinstance(dim, (int, float)):
			self.dim = dim
		else:
			raise ValueError('not numeric')
		if self.dim>self.p:
			raise ValueError('dim bigger than p')
		if (alpha>=0 and alpha<=1):
			self.alpha = alpha
		else:
			raise ValueError('not between 0 and 1')

	def standardize(self):
		medias = self.data.mean(axis=0)
		desv = self.data.std(axis=0)
		self.data_st = (self.data-medias)/desv
		return self.data_st

	def SVD(self,niter=5,state=None,std=True):
		if std==True:
			self.data = self.standardize()
		U, Sigma, VT = randomized_svd(self.data, n_components=self.dim,n_iter=niter,random_state=state)
		return U, Sigma, VT

	def Inertia(self):
		U, Sigma, VT = self.SVD()
		self.EV = np.power(Sigma,2)
		self.Inertia = self.EV/np.sum(self.EV) * 100

	def Contributions(self):
		U, Sigma, VT = self.SVD()
		R = U.dot(np.diag(Sigma[:self.dim]))
		C = np.transpose(VT).dot(np.diag(Sigma[:self.dim]))

		sf = np.sum(np.power(X_st,2),axis=1)
		cf = np.zeros((n,dim))
		for k in range(0,dim):
			cf[:,k] = np.power(R[:,k],2)*100/sf

		sc = np.sum(np.power(X_st,2),axis=0)
		cc = np.zeros((p,dim))

		for k in range(0,dim):
			cc[:,k] = np.power(C[:,k],2)*100/sc

		self.RowContributions = cf
		self.ColContributions = cc

