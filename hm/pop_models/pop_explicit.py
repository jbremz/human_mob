from hm.pop_models.population import pop_distribution
import numpy as np

class explicit(pop_distribution):
	'''
	Population distribution which takes explicit coordinates and population distriubtion

	'''

	def __init__(self, locCoords, popDist, locArea = None):
		self.locCoords = np.array(locCoords)
		self.popDist = np.array(popDist)
		self.locArea = np.array(locArea)
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
	
	def loc_area(self):
		'''
		Returns the surface area distribution
		'''
		
		return self.locArea