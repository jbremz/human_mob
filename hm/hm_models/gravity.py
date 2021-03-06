# from hm.utils.utils import disp
import numpy as np
from .base_model import mob_model

class gravity(mob_model):
	'''
	The gravity human mobility model

	'''

	def __init__(self, pop, alpha, beta, gamma, **kwargs):
		super().__init__(pop)
		kwargs.setdefault('exp', False)
		self.alpha = alpha # population i exponent
		self.beta = beta # population j exponent
		self.gamma = gamma # distance exponent
		self.pop = pop # population object
		self.exp = kwargs['exp'] # True if exponential decay function is used, False if power decay is used
		self.K = self.K() # calulates the K normalisation factors
		
	def K(self):
		'''
		The normalisation constant K

		'''
		k_s = []
		for i in range(self.pop.size):
			factor = []
			for j in range(self.pop.size):
				if j != i:
					factor.append(self.pop.popDist[j]*self.f(self.pop.DM[i,j]))
			k_s.append(1./np.sum(factor))
		return k_s

	def f(self, r):
		'''
		The distance decay function
		'''

		if self.exp:
			return np.exp(-self.gamma*r)
		else:
			return r**(-self.gamma)

	def flux(self, i, j, probs=True):
		'''
		Takes the indices of two locations and returns the flux between them
		'''
		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		r = pop.DM[i, j]

		# Probabilities
		if probs:
			n = self.K[i] * popj*self.f(r)

		# Flows
		if not probs:
			n = popi * self.K[i] * popj*self.f(r)

		return n

