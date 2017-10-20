from hm.pop_models.population import pop_distribution
import numpy as np

class explicit(pop_distribution):
	'''
	Population distribution which takes explicit coordinates and population distriubtion

	'''

	def __init__(self, locCoords, popDist):
		self.locCoords = np.array(locCoords)
		self.popDist = np.array(popDist)
		super().__init__()

	def loc_dist(self):
		'''
		Returns the 2D population distribution

		'''

		return self.locCoords

	def pop_dist(self):
		'''
		Returns the population distribution

		'''

		return self.popDist