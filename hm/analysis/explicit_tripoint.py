from hm.pop_models.pop_random import random as pop_random
from hm.hm_models.gravity import gravity
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
import copy

def epsilon(x,y,N, size=1., ib=True):
	'''
	Takes the x and y displacements defined in the tripoint problem and returns
	the error between treating the satellite locations as one and as separate
	ib = True --> calculate Tib 
	ib = False --> calulate Tbi (i.e. flow in opposite direction)

	'''
	p = pop_random(N)

	loci = [0.5+x/2., 0.5]
	locj = [0.5-x/2., 0.5-y/2.]
	lock = [0.5-x/2., 0.5+y/2.]
	locb = [0.5-x/2., 0.5]

	sizeb = 2*size # TODO change to non-uniform size

	p3 = copy.deepcopy(p) # the tripoint arrangement
	p2 = copy.deepcopy(p) # the two-point arrangement

	p3.locCoords = np.insert(p3.locCoords, 0, np.array([loci, locj, lock]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)
	p3.popDist = np.insert(p3.popDist, 0, np.array([size, size, size]), axis=0)
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, sizeb]), axis=0)

	g3 = gravity(p3, 1, 1, 2) # TODO Gamma=2, try other exponents and other functions?
	g2 = gravity(p2, 1, 1, 2)

	if ib: # From i to the satellites
		g2Flux = g2.flux(0,1)
		eps = (g2Flux - (g3.flux(0,1)+g3.flux(0,2)))/g2Flux

	# TODO This doesn't work
	else: # from the satellites to i 
		g2Flux = g2.flux(1,0)
		eps = (g2Flux - (g3.flux(1,0)+g3.flux(2,0)))/g2Flux

	return np.array(eps)



def anaTP(xmin, xmax, ymin, ymax, n, N, ib=True):
	'''
	Finds values of epsilon in an nxn 2D sample space for x and y at fixed N random locations

	'''

	x = np.linspace(xmin, xmax, n)
	y = np.linspace(ymin, ymax, n)

	xy = np.array(np.meshgrid(x, y)).T # 2D Sample Space
	xy = np.swapaxes(xy,0,1)

	epsVals = np.zeros((n,n))

	for j, row in enumerate(xy):
		for i, pair in enumerate(row):
			epsVals[j][i] = epsilon(pair[0], pair[1], N, ib=ib)

	nanMask = np.isnan(epsVals)

	epsVals[nanMask] = 1

	epsVals = np.flip(epsVals, 0) # make the y axis ascend

	xticks = np.around(x*np.sqrt(N), 2)
	yticks = np.flip(np.around(y*np.sqrt(N), 2), 0)

	ax = sns.heatmap(epsVals, xticklabels=xticks, yticklabels=yticks, square=True) # TODO contour plot

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	ax.set_xlabel(r'$x \sqrt{N}$')
	ax.set_ylabel(r'$y \sqrt{N}$')

	plt.show()

	return





