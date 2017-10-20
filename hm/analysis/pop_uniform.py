from hm.pop_models.population import pop_distribution
import numpy as np

class uniform(pop_distribution):
	'''
	Uniform population distribution of N locations in 2D space (0<x<1, 0<y<1), and uniform population distribution of length N

	'''

	def __init__(self, N, uniform=True):
		self.N = N
		self.uniform = uniform
		super().__init__()

	def pop_dist(self):
		'''
		Returns the population distribution for N locations

		TODO: non-uniform

		'''
		if self.uniform:
			return np.ones(self.N)

	def loc_dist(self):
		'''
		Returns the uniform 2D population distribution: N locations (0<x<1, 0<y<1)

		'''
		interval = 1./(np.sqrt(self.N)-1)
		x = []
		y = []
		coords = []
		for i in np.arange(0., 1.1, interval):
			x.append(i)
			y.append(i)
		for i in np.arange(np.sqrt(self.N)):
			for j in np.arange(np.sqrt(self.N)):
				coords.append([x[int(i)], y[int(j)]])

		return np.array(coords)