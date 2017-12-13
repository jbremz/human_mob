from hm.utils.utils import disp
import numpy as np

class mob_model:
	'''
	Base human mobility model class
	'''
	def __init__(self, pop):
		self.pop = pop # the population object

	def ODM(self, probs=False):
		'''
		returns the predicted origin-destination flow matrix for the population distribution

		'''
		pop = self.pop
		e = len(pop.locCoords)
		m = np.zeros((pop.size, pop.size)) # original OD matrix to be filled with fluxes

		for i in range(pop.size):
			for j in range(pop.size):
				if i != j: 
					f = self.flux(i,j,probs=probs)
					m[i][j] = f

		return m