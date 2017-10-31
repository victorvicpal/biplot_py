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

	def SVD(self,niter=5,state=None,std=True):
		if std==True:
			self.standardize()
			data_int = self.data_st
		else:
			data_int = self.data
		
		U, Sigma, VT = randomized_svd(data_int, n_components=self.dim,n_iter=niter,random_state=state)
		return U, Sigma, VT

	def Inertia(self,std=True):
		U, Sigma, VT = self.SVD(std=std)
		self.EV = np.power(Sigma,2)
		self.Inertia = self.EV/np.sum(self.EV) * 100

	def Contributions(self,std=True):
		if std==True:
			self.standardize()
			data_int = self.data_st
		else:
			data_int = self.data

		U, Sigma, VT = self.SVD(std=std)
		R = U.dot(np.diag(Sigma[:self.dim]))
		C = np.transpose(VT).dot(np.diag(Sigma[:self.dim]))

		sf = np.sum(np.power(data_int,2),axis=1)
		cf = np.zeros((self.n,self.dim))
		for k in range(0,self.dim):
			cf[:,k] = np.power(R[:,k],2)*100/sf

		sc = np.sum(np.power(data_int,2),axis=0)
		cc = np.zeros((self.p,self.dim))

		for k in range(0,self.dim):
			cc[:,k] = np.power(C[:,k],2)*100/sc

		self.RowContributions = cf
		self.ColContributions = cc

