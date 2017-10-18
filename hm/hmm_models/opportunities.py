from hm.utils.utils import disp
import numpy as np
from .base_model import mob_model

class opportunities(mob_model):
	'''
	The intervening opportunities model
	'''
	def __init__(self, pop, gamma):
		super().__init__(pop)
		self.gamma = gamma

	def norm_factor(self, i, j):
		pop = self.pop
		to_sum = []
		for k in range(pop.size):
			popk = pop.popDist[k]
			popSik = pop.s(i, k)
			if k != j:
				a = np.exp((-self.gamma*popSik))-(np.exp(-self.gamma*(popSik+popk)))
				to_sum.append(a)
		return sum(to_sum)

	def flux(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the average flux from i to j
		'''
		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		popSij = pop.s(i, j)
		a = np.exp((-self.gamma*popSij)-(np.exp(-self.gamma*(popSij)+popj)))
		n = self.norm_factor(i, j)*popi*a
		return n