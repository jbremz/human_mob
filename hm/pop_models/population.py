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
		closer_pop = []
		for loc in range(self.size):
			if disp(self.locCoords[i], self.locCoords[loc]) < self.r(i, j):
				if loc != i:
					if loc != j:
						closer_pop.append(self.popDist[loc])

		return sum(closer_pop)

	def r(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the distance between them
		'''
		r = disp(self.locCoords[i], self.locCoords[j])
		return r

	def M(self):
		'''
		Returns the total sample population

		'''
		return np.sum(np.array(self.popDist))
