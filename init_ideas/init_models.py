
from utils import disp
import numpy as np

class simple_gravity:
	'''
	The simple gravity model

	'''
	def __init__(self, popDist, locCoords, beta, K):
		self.beta = beta # inverse distance exponent
		self.popDist = popDist # list of populations in each location
		self.K = K # fitting parameter
		self.locCoords = locCoords # an array of coordinates of the locations 

	def flux(self, i, j):
		'''
		Takes the indices of two locations and returns the flux between them

		'''
		popi, popj = self.popDist[i], self.popDist[j]
		r = disp(self.locCoords[i], self.locCoords[j])
		n = self.K * (popi*popj)/r**self.beta

		return n 

	def ODM(self):
		'''
		Returns the predicted origin-destination flow matrix for the population distribution

		'''
		size = len(self.locCoords)
		m = np.zeros((size, size)) # original OD matrix to be filled with fluxes

		for i in range(size):
			for j in range(i+1,size):
				f = self.flux(i,j)
				print([i,j], f)
				m[i][j], m[j][i] = f, f # symmetrical

		return m





