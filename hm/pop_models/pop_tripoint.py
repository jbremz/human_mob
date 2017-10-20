from hm.pop_models.population import pop_distribution
import numpy as np

class tripoint(pop_distribution):
	'''
	Population distribution which considers pair of locations aligned on y-axis (ydisp apart) at certain x-displacement (xdisp) from a third point.
	Adds N random background locations

	'''

	def __init__(self, N, xdisp, ydisp, uniformSize=True):
		self.N = N
		self.xdisp = xdisp
		self.ydisp = ydisp
		self.uniformSize = uniformSize # uniform population distribution between locations
		super().__init__()

	def loc_dist(self):
		'''
		Returns the 2D population distribution: N locations (0<x<1, 0<y<1)

		'''

		loci = [0.5+self.xdisp/2., 0.5]
		locj = [0.5-self.ydisp/2., 0.5-self.ydisp/2.]
		lock = [0.5+self.ydisp/2., 0.5+self.ydisp/2.]

		locs = np.random.rand(self.N-3,2)
		ijk = np.array([loci, locj, lock])
		locs = np.insert(ijk, 3, locs, axis=0)

		return locs

	def pop_dist(self):
		'''
		Returns the population distribution for N locations

		TODO: non-uniform

		'''
		if self.uniformSize:
			return np.ones(self.N)

