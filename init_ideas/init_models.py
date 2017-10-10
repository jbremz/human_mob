


class simple_gravity:
	'''
	The simple gravity model

	'''
	def __init__(self, popDist, locCoords, beta, K):
		self.beta = beta # inverse distance exponent
		self.popDist = popDist # list of populations in each location
		self.K = K # fitting parameter

	def flux(self, i, j):
		'''
		Takes the indices of two locations and returns the flux between them

		'''
		popDist = self.popDist
		beta = self.beta
		K = self.K

		popi, popj = popDist[i], popDist[j]
		r = 

		n = K * (popi*popj)/






