from bipfun import *

class CanonicalBip(object):
	'''
	Canonical Biplot (Vicente-Villardon)
	'''

	def __init__(self, data,dim,GroupNames,y):
		self.data = data
		if isinstance(dim, (int, float)):
			self.dim = dim
		else:
			raise ValueError('not numeric')
		if self.dim>self.data.shape[1]:
			raise ValueError('dim bigger than p')
		if isinstance(GroupNames, (list)):
			self.GroupNames = GroupNames
		else:
			raise ValueError('not numeric')

		if isinstance(y, list):
			self.target = y
		else:
			raise ValueError('not list')

	def CanonicalBip(self,method=None,niter=5,state=None):
		self.data_st = standardize(self.data,meth=method)
		data_std = self.data_st
		g = len(self.GroupNames)
		n = self.data.shape[0]
		m = self.data.shape[1]
		r = np.min(np.array([g - 1, m]))

		#Groups to Binary
		Z = Factor2Binary(self.target)
		ng = Z.sum(axis=0)
		S11 = (Z.T).dot(Z)
		Xb = np.linalg.inv(S11).dot(Z.T).dot(data_std)
		B = (Xb.T).dot(S11).dot(Xb)
		S = (data_std.T).dot(data_std) - B
		Y = np.power(S11,0.5).dot(Xb).dot(matrixsqrt(S,self.dim,inv=True))

		U, Sigma, VT = SVD(Y,self.dim,niter,state)

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
		V = data_std.dot(B)
		self.Ind_Coord = V

		sct = np.diag((V.T).dot(V))
		sce = np.diag((J.T).dot(S11).dot(J))
		scr = sct -sce
		fs = (sce/(g - 1))/(scr/(n - g))

		#eigenvectors
		vprop = Sigma[:r]
		self.vprop = vprop
		#Inertia
		iner = (np.power(vprop,2)/(np.power(vprop,2).sum()))*100
		self.inert = iner

		lamb = np.power(vprop,2)
		pill = 1/(1 + lamb)
		pillai = np.linalg.det(np.diag(pill))
		glh = g - 1
		gle = n - g
		t = np.sqrt((np.power(glh,2) * np.power(m,2) - 4)/(np.power(m,2) + np.power(glh,2) - 5))
		w = gle + glh - 0.5 * (m + glh + 1)
		df1 = m * glh
		df2 = w * t - 0.5 * (m * glh - 2)

		Wilksf = (1 - np.power(pillai,1/t))/(np.power(pillai,1/t)) * (df2/df1)
		Wilksp = stats.f.pdf(Wilksf, df1, df2)
		self.Wilks = [['f-val',Wilksf],['p-val',Wilksp]]

		falfau = stats.t.ppf(1 - (0.025), (n - g))
		falfab = stats.t.ppf(1 - (0.025/(g * m)), (n - g))
		falfam = np.sqrt(stats.f.ppf(1 - 0.05, m, (n - g - m + 1)) * (((n - g) * m)/(n - g - m + 1)))
		falfac = 2.447747

		UnivRad = falfau * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
		BonfRad = falfab * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
		MultRad = falfam * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
		ChisRad = falfac * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)

		self.Radius = [['Uni',UnivRad],['Bonf',BonfRad],['Mult',MultRad],['Chis',ChisRad]]

