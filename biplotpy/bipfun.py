import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy import stats

def standardize(data,meth=None):
	global data_stan
	if meth == None:
		data_stan = data
	if meth == 1:
		medias = data.mean(axis=0)
		desv = data.std(axis=0)
		data_stan = (data-medias)/desv
	return data_stan

def SVD(M,dimen,niter=5,state=None):
	U, Sigma, VT = randomized_svd(M, n_components=dimen,n_iter=niter,random_state=state)
	return U, Sigma, VT

def Inertia(M,dimen,niter=5,state=None):
	U, Sigma, VT = SVD(M,dimen,niter,state)

	EV = np.power(Sigma,2)
	Inert = EV/np.sum(EV) * 100
	return EV, Inert

def Contributions(M,dimen,n,p,niter=5,state=None):
	U, Sigma, VT = SVD(M,dimen,niter,state)

	R = U.dot(np.diag(Sigma[:dimen]))
	C = np.transpose(VT).dot(np.diag(Sigma[:dimen]))

	sf = np.sum(np.power(M,2),axis=1)
	cf = np.zeros((n,dimen))
	for k in range(0,dimen):
		cf[:,k] = np.power(R[:,k],2)*100/sf

	sc = np.sum(np.power(M,2),axis=0)
	cc = np.zeros((p,dimen))
	for k in range(0,dimen):
		cc[:,k] = np.power(C[:,k],2)*100/sc

	return cf, cc

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
	U, Sigma, VT = randomized_svd(M, n_components=dim, n_iter=5, random_state=None)
	nz = Sigma > tol
	if inv==True:
		S12 = U.dot(np.diag(1/np.sqrt(Sigma[nz]))).dot(VT[nz,:])
	else:
		S12 = U.dot(np.diag(np.sqrt(Sigma[nz]))).dot(VT[nz,:])
	return S12
