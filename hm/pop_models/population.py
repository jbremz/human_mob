from hm.utils.utils import disp
import numpy as np

class pop_distribution:
	'''
	Base population distribution class. Contains general methods for population metrics.

	'''

	def __init__(self):
		self.locCoords = self.loc_dist()
		self.popDist = self.pop_dist()
		self.size = len(self.locCoords)
		
	def s(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the population in a circle of radius r 
		(r = distance between i and j) centred on i 
		'''
		r = disp(self.locCoords[i], self.locCoords[j])
		closer_pop = []
		for loc in range(self.size):
			if disp(self.locCoords[i], self.locCoords[loc]) <= r:
				if loc != i:
					if loc != j:
						closer_pop.append(self.popDist[loc])
						
		return sum(closer_pop)

	def M(self):
		'''
		Returns the total sample population

		'''
		return np.sum(np.array(self.popDist))

def random(N, uniform=True):
	'''
	Returns random distribution of N locations in 2D space (0<x<1, 0<y<1), and uniform population distribution of length N

	TODO: random size distribution also?

	'''
	return np.random.rand(N,2), np.ones(N)
