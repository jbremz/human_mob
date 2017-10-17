from hm.utils.utils import disp
import numpy as np

class pop_distribution:
	def __init__(self, popDist, locCoords):
		self.locCoords = locCoords
		self.popDist = popDist
		self.size = len(self.locCoords)
		
	def s(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the population in a circle of radius r 
		(= distance between two locations) centred on i 
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