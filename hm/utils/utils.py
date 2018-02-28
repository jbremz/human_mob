import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import time


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

def sum4s(m):
	'''
	Splits NxN matrix m into 2x2 submatrices (chequerboard) and returns the sum of all elements in each submatrix as an (N/2)x(N/2) matrix  

	'''
	Lnew = int(len(m)/2)
	return view_as_windows(m, (2,2),step=2).reshape(int(m.size/4),4).sum(axis=1).reshape(Lnew,Lnew)

def gamma_est(S, exp=False):
	'''
	Takes the average population unit area <S> in m^2 and returns an estimate for the gamma exponent as proposed by Lenormand et al. 2012 

	'''

	if exp:
		return 0.3 * (S/1000000)**(-0.18) / 1000 # divide by 1000 to account for the change in distance units
	else:
		return 1.4 * (S/1000000)**(0.11) 

def time_label():
	'''
	Returns current datetime string for labelling plots/data 

	'''
	t = time.localtime()
	return time.strftime('%b-%d-%Y_%H%M', t)



