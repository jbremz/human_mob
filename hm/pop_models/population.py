from hm.utils.utils import disp
import numpy as np
from scipy.spatial.distance import pdist, squareform

class pop_distribution:
	'''
	Base population distribution class. Contains general methods for population metrics.

	'''

	def __init__(self):
		self.locCoords = self.loc_dist()
		self.popDist = self.pop_dist()
		self.size = len(self.locCoords)
		self.M = self.M_()
		self.DM = self.distance_matrix()
		self.locArea = self.loc_area() #this function does not give the right 
									# output for pop_random

	def s(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the population in a circle of radius r
		(r = distance between i and j) centred on i

		'''
		ds = self.DM[i]
		r = self.DM[i][j]

		closer_pops = np.compress(ds<r, self.popDist)

		return np.sum(closer_pops) - self.popDist[i]

	def flat_distance(self):
		'''Returns the condensed distance matrix as a 1d numpy array.''' 
		
		return pdist(self.locCoords)
	
	def distance_matrix(self):
		self.flat_DM = self.flat_distance()
		return squareform(pdist(self.locCoords))

	def r(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the distance between them
		'''
		r = disp(self.locCoords[i], self.locCoords[j])
		return r

	def M_(self):
		'''
		Returns the total sample population

		'''
		return np.sum(np.array(self.popDist))