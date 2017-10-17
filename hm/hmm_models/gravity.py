from hm.utils.utils import disp
import numpy as np
from .base_model import mob_model

class gravity(mob_model):
	'''
	The gravity human mobility model

	'''

	def __init__(self, pop, alpha, beta, gamma, K, **kwargs):
		super().__init__(pop)
		kwargs.setdefault('exp', False)
		self.alpha = alpha # population i exponent
		self.beta = beta # population j exponent
		self.gamma = gamma # distance exponent
		self.K = K # fitting parameter
		self.exp = kwargs['exp'] # True if exponential decay function is used, False if power decay is used

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
		r = disp(pop.locCoords[i], pop.locCoords[j])
		n = self.K * ((popi**self.alpha)*(popj**self.beta))*self.f(r)
		return n