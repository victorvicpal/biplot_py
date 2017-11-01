#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy import stats

class biplotpy:
	'''
	Gabriel Biplots
	Canonical Biplot
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

	def SVD(self,std=True,niter=5,state=None):
		if std==True:
			self.standardize()
			data_int = self.data_st
		else:
			data_int = self.data
		
		U, Sigma, VT = randomized_svd(data_int, n_components=self.dim,n_iter=niter,random_state=state)
		return U, Sigma, VT

	def Inertia(self,std=True):
		if std==True:
			U, Sigma, VT = self.SVD(std=True)
		else:
			U, Sigma, VT = self.SVD(std=False)

		self.EV = np.power(Sigma,2)
		self.Inertia = self.EV/np.sum(self.EV) * 100

	def Contributions(self,std=True):
		if std==True:
			U, Sigma, VT = self.SVD(std=True)
			data_int = self.data_st
		else:
			U, Sigma, VT = self.SVD(std=False)
			data_int = self.data

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

	def biplot(self,std=True):
		if std==True:
			U, Sigma, VT = self.SVD(std=True)
			self.Contributions(std=True)
			self.Inertia(std=True)
		else:
			U, Sigma, VT = self.SVD(std=False)
			self.Contributions(std=False)
			self.Inertia(std=False)

		R = U.dot(np.diag(Sigma[:self.dim]))
		C = np.transpose(VT).dot(np.diag(Sigma[:self.dim]))

		R = R.dot(np.diag(np.power(Sigma,self.alpha)))
		C = C.dot(np.diag(np.power(Sigma,1-self.alpha)))

		sca = np.sum(np.power(R,2))/self.n
		scb = np.sum(np.power(C,2))/self.p
		scf = np.sqrt(np.sqrt(scb/sca))

		self.R = R*scf
		self.C = C/scf

	def plot_bip(self,col_names,labels,std=True,dim1=1,dim2=2,fisize=None,warrow=0.07,fosize=20):
		if fisize==None:
			fisize = self.R[:,[dim1-1,dim2-1]].max(axis=0).max()

		fig = plt.figure(figsize=(fisize,fisize))
		ax1 = fig.add_subplot(111)

		ax1.scatter(self.R[:,dim1-1],self.R[:,dim2-1],c=labels)
		for i in range(0,self.C.shape[0]):
			ax1.arrow(0,0,self.C[i,dim1-1],self.C[i,dim2-1],width=warrow)
			ax1.text(self.C[i,0],self.C[i,1],col_names[i],fontsize=fosize)
		plt.show()

	def CanonicalBip(self,GroupNames,y,std=True):
		if isinstance(GroupNames, (list)):
			self.GroupNames = GroupNames
		else:
			raise ValueError('not numeric')

		if isinstance(y, (np.ndarray)):
			self.target = y
		else:
			raise ValueError('not numeric')

		if std==True:
			self.standardize()
			data = self.data_st
		else:
			data = self.data

		g = len(GroupNames)
		n = data.shape[0]
		m = data.shape[1]
		r = np.min(np.array([g - 1, m]))

		def Factor2Binary(y,Name = None):
			if Name == None:
				Name = "C"
			ncat = len(list(set(y)))
			n = len(y)
			Z = pd.DataFrame(0, index=np.arange(len(y)), columns=list(set(y)))
			for col in Z.columns:
				for i in range (0,n):
					if y[i] == col:
						Z[col].iloc[i] = 1
			return Z

		def matrixsqrt(M,dim,tol=np.finfo(float).eps,inv=True):
			U, Sigma, VT = randomized_svd(M, n_components=self.dim, n_iter=5, random_state=None)
			nz = Sigma > tol
			if inv==True:
				S12 = U.dot(np.diag(1/np.sqrt(Sigma[nz]))).dot(VT[nz,:])
			else:
				S12 = U.dot(np.diag(np.sqrt(Sigma[nz]))).dot(VT[nz,:])
			return S12

		#Groups to Binary
		Z = Factor2Binary(y)
		ng = Z.sum(axis=0)
		S11 = (Z.T).dot(Z)
		Xb = np.linalg.inv(S11).dot(Z.T).dot(data)
		B = (Xb.T).dot(S11).dot(Xb)
		S = (data.T).dot(data) - B
		Y = np.power(S11,0.5).dot(Xb).dot(matrixsqrt(S,self.dim,inv=True))

		U, Sigma, VT = randomized_svd(Y, n_components=self.dim, n_iter=5, random_state=None)

		#Variable_Coord
		H = matrixsqrt(S,self.dim,inv=False).dot(np.transpose(VT[0:r,:]))
		self.Var_Coord = H
		#Canonical_Weights
		B = matrixsqrt(S,self.dim,inv=True).dot(np.transpose(VT[0:r,:]))
		self.Can_Weights = B
		#Group_Coord
		J = Xb.dot(B)
		self.Group_Coord = J
		#Individual_Coord
		V = data.dot(B)
		self.Ind_Coord = V
