from hm.utils.utils import disp
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

	def K(self, i):
		'''
		The normalisation constant K

		'''
		factor = []
		for j in range(self.pop.size):
			if j != i:
				factor.append(self.pop.pop_dist()[j]*self.f(self.pop.r(i,j)))
		return 1./sum(factor)

	def f(self, r):
		'''
		The distance decay function
		'''

		if self.exp:
			return np.exp(-self.gamma*r)
		else:
			return r**(-self.gamma)

	def flux(self, i, j):
		'''
		Takes the indices of two locations and returns the flux between them
		'''

		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		r = pop.r(i, j)
		n = self.K(i) * ((popi**self.alpha)*(popj**self.beta))*self.f(r)
		return n