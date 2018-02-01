import ClassicBip
import CanonicalBip
import pandas
import numpy
import itertools

class Feature_selection(object):
	'''
	Feature selection for classification problems
	'''

	def __init__(self, bip, target, thr_dis = 75, thr_corr = 0.89, type_cor = "global"):
		if isinstance(bip, ClassicBip.ClassicBip):
			__type__ = "Classic"
			self.bip = bip
		elif isinstance(bip, CanonicalBip.CanonicalBip):
			__type__ = "Canonical"
			self.bip = bip
		else:
			raise ValueError('Undefined biplotpy class')

		if isinstance(target, pandas.core.series.Series):
			if isinstance(list(set([type(el) for el in target]))[0], (int,float)):
				self.y = numpy.array(target)
		elif isinstance(target, numpy.ndarray):
			self.y = target
		else:
			raise ValueError('Nor ndarray numpy nor series pandas type')

		if isinstance(thr_dis, (float,int)) == False:
			raise ValueError('Nor ndarray numpy nor series pandas type')
		elif thr_dis > 100 :
			raise ValueError('thr_dis must be between 25 and 100')
		elif thr_dis < 0 :
			raise ValueError('thr_dis must be positive')

		if __type__ == "Classic":
			Project = bip.RowCoord.dot(bip.ColCoord.T)
			C = bip.ColCoord
		elif __type__ == "Canonical":
			Project = bip.Ind_Coord.dot(bip.Var_Coord.T)
			C = bip.Var_Coord

		# Positive rescalation of projections

		v_min = numpy.array([abs(el) if el < 0 else el for el in Project.min(axis=0)])

		for i, proj in enumerate(Project.T):
			Project[:,i] = proj + v_min[i]
		
		classes = numpy.unique(target)

		# Tracking class index

		IND = []
		for cl in classes:
			ind_class = []
			for i, el in enumerate(target):
				if el == cl:
						ind_class.append(i)
			IND.append(ind_class)

		# Number of combinations

		num_c = int(len(classes)*(len(classes)-1)/2)

		Disc = numpy.zeros((bip.data.shape[1], num_c))

		comb = numpy.array(list(itertools.combinations(classes,r=2)))

		# Disc vectors

		for i, cmb in enumerate(comb):
			Disc[:,i] = abs(Project[IND[cmb[0]]].mean(axis=0) - Project[IND[cmb[1]]].mean(axis=0))

		self.Disc = Disc

		# Drop correlated variables

		POS = []
		for v in Disc.T:
			for i, el in enumerate(v):
				if el > numpy.percentile(v, thr_dis):
					POS.append(i)
		POS = list(set(POS))

		self.POS_not = POS


		if type_cor == "global":
			Corr_matr = numpy.tril(numpy.corrcoef(bip.data[:,POS].T), -1)
		elif type_cor == "coord":
			Corr_matr = numpy.tril(numpy.corrcoef(C[POS,:]), -1)
		elif type_cor == "discr":
			Corr_matr = numpy.tril(numpy.corrcoef(Disc[POS,:]), -1)
		else:
			raise ValueError('type_cor must be "global", "coord" or "discr"')

		self.Corr_matr = Corr_matr

		### Correlation threshold (23/01/2018)

		#pos_corr = numpy.where(Corr_matr > thr_corr)
		#disc_vect = Disc.sum(axis = 1)

		#self.disc_vect = disc_vect

		#del_el = []
		#if pos_corr:
		#	for i in range(len(pos_corr[0])):
		#		ind = [pos_corr[0][i],pos_corr[1][i]]
		#		ind_del = []
		#		if ((ind[0] in POS) and (ind[1] in POS)):
		#			a = numpy.array([disc_vect[ind[0]],disc_vect[ind[1]]])
		#			ind_del.append(POS.index(pos_corr[ numpy.argwhere(a.min() == a)[0][0] ][0]))

		### Correlation threshold (01/02/2018)


		pos_corr = numpy.where(Corr_matr > thr_corr)
		disc_vect = Disc[POS,:].sum(axis = 1)

		ind_del = []
		if pos_corr:
			for i in range(len(pos_corr[0])):
				if disc_vect[pos_corr[0][i]] > disc_vect[pos_corr[1][i]]:
					ind_del.append(pos_corr[1][i])
				else:
					ind_del.append(pos_corr[0][i])


		ind_del = list(set(ind_del))
		if ind_del:
			POS = [el for i, el in enumerate(POS) if i not in ind_del]

		self.var_sel = list(numpy.array(bip.col_names)[POS])



