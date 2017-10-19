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

		return self.locCoords

	def pop_dist(self):

		return self.popDist