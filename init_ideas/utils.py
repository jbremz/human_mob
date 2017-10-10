import numpy as np

def disp(loci, locj):
	'''
	Takes the coordinates of locations i and j and returns the euclidean distance between them

	'''
	s = locj - loci
	return np.sqrt(np.dot(s,s))