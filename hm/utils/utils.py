import numpy as np
import matplotlib.pyplot as plt

def disp(loci, locj):
	'''
	Takes the coordinates of locations i and j and returns the euclidean distance between them

	'''
	s = locj - loci
	return np.sqrt(np.dot(s,s))

def plot_pop(population, show=True, **kwargs):
	'''
	Takes a population object and plots the locations

	TODO: area proportional to population? 

	'''
	coords = population.locCoords
	plt.scatter(coords[:,0], coords[:,1], s=4)

	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')

	if show:
		plt.show()

	return


