from hm.pop_models.population import pop_distribution
import numpy as np

class random(pop_distribution):
	'''
	Random population distribution of N locations in 2D space (0<x<1, 0<y<1), and uniform population distribution of length N

	TODO: random size distribution also?

	'''

	def __init__(self, N, uniformSize=True):
		self.N = N
		self.uniformSize = uniformSize # uniform population distribution between locations
		super().__init__()

	def pop_dist(self):
		'''
		Returns the population distribution for N locations

		TODO: non-uniform

		'''
		if self.uniformSize:
			return np.ones(self.N)

	def loc_dist(self):
		'''
		Returns the random 2D population distribution: N locations (0<x<1, 0<y<1)

		'''
		return np.random.rand(self.N,2)