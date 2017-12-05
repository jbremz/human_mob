from hm.pop_models.population import pop_distribution
import numpy as np

class random(pop_distribution):
	'''
	Random population distribution of N locations in 2D space (0<x<1, 0<y<1), and uniform population distribution of length N

	TODO: random size distribution also?

	'''

	def __init__(self, N, uniformSize=True, **kwargs):
		self.N = N
		self.uniformSize = uniformSize # uniform population distribution between locations
		kwargs.setdefault('seed', False)
		self.seed = kwargs['seed']
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
		if type(self.seed) is bool:
			return np.random.rand(self.N,2)

		# for repeatable results
		else: 
			np.random.seed(self.seed)
			return np.random.rand(self.N,2)

		