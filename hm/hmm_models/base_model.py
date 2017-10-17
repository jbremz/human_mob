from hm.utils.utils import disp
import numpy as np

class mob_model:
	'''
	Base human mobility model class
	'''
	def __init__(self, pop):
		self.pop = pop # the population object

	def ODM(self):
		'''
		returns the predicted origin-destination flow matrix for the population distribution

		'''
		pop = self.pop
		e = len(pop.locCoords)
		m = np.zeros((pop.size, pop.size)) # original OD matrix to be filled with fluxes

		for i in range(pop.size):
			for j in range(i+1,pop.size):
				f = self.flux(i,j)
				m[i][j], m[j][i] = f, f # symmetrical

# Test Data

popDist = [3,5,6,2,1]
locCoords = np.array([[0,0],[1,3],[4,7],[-3,-5],[6,-9]])
beta = 0.2
K = 10

