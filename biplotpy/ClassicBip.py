from bipfun import *

class ClassicBip(object):
	'''
	Gabriel Biplots
	'''

	def __init__(self, data,dim,alpha = 1):
		self.data = data
		if isinstance(dim, (int, float)):
			self.dim = dim
		else:
			raise ValueError('not numeric')
		if self.dim > self.data.shape[1]:
			raise ValueError('dim bigger than p')
		if (alpha>=0 and alpha<=1):
			self.alpha = alpha
		else:
			raise ValueError('not between 0 and 1')

	def biplot(self,method=None,niter=5,state=None):
		self.data_st = standardize(self.data,meth=method)
		U, Sigma, VT = SVD(self.data_st,self.dim,niter,state)

		self.EV, self.Inert = Inertia(self.data_st,self.dim,niter,state)
		self.RowCont, self.ColCont = Contributions(self.data_st,self.dim,self.data.shape[0],self.data.shape[1],niter,state)

		R = U.dot(np.diag(Sigma[:self.dim]))
		C = np.transpose(VT).dot(np.diag(Sigma[:self.dim]))

		R = R.dot(np.diag(np.power(Sigma,self.alpha)))
		C = C.dot(np.diag(np.power(Sigma,1-self.alpha)))

		sca = np.sum(np.power(R,2))/self.data.shape[0]
		scb = np.sum(np.power(C,2))/self.data.shape[1]
		scf = np.sqrt(np.sqrt(scb/sca))

		self.RowCoord = R*scf
		self.ColCoord = C/scf