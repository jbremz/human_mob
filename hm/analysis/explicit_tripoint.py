from hm.pop_models.pop_random import random as pop_random
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.utils.utils import plot_pop
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
import copy

def createPops(x,y,N,size,seed):
	'''
	Returns the tripoint and two-point population objects as well as location of i and location of b

	'''
	if type(seed) is bool:
		p = pop_random(N)

	else: # use seed for random population distribution so that it can be replicated
		p = pop_random(N, seed=seed)

	loci = [0.5+x/2., 0.5]
	locj = [0.5-x/2., 0.5-y/2.]
	lock = [0.5-x/2., 0.5+y/2.]
	locb = [0.5-x/2., 0.5]

	p3 = copy.deepcopy(p) # the tripoint arrangement
	p2 = copy.deepcopy(p) # the two-point arrangement

	p3.locCoords = np.insert(p3.locCoords, 0, np.array([loci, locj, lock]), axis=0)
	p3.popDist = np.insert(p3.popDist, 0, np.array([size, size, size]), axis=0)

	return p2, p3, loci, locb

def tilde_m(size, mobObj):
	'''
	Takes pop size and mobility model object and returns the corrected tilde m for the size

	# TODO: change to non-uniform size
	'''
	return 2*size - size*(mobObj.flux(1,2) - mobObj.flux(2,1))

def epsilon(mobObj2, mobObj3, ib=True):
	'''
	Takes two-point and tri-point mobility model objects and returns epsilon

	ib = True --> calculate Tib 
	ib = False --> calulate Tbi (i.e. flow in opposite direction)

	'''

	if ib: # From i to the satellites
		mob2Flux = mobObj2.flux(0,1)
		eps = (mob2Flux - (mobObj3.flux(0,1)+mobObj3.flux(0,2)))/mob2Flux

	else: # from the satellites to i 
		mob2Flux = mobObj2.flux(1,0)
		eps = (mob2Flux - (mobObj3.flux(1,0)+mobObj3.flux(2,0)))/mob2Flux

	return eps


def epsilon_g(x,y,N, size=1., ib=True, seed=False, tildeM=False, gamma=2):
	'''
	Takes the x and y displacements defined in the tripoint problem and returns
	the error between treating the satellite locations as one and as separate

	seed = False --> use completely random pop dist
	seed = int --> recreate a previous population distribution

	'''
	p2, p3, loci, locb = createPops(x,y,N,size,seed)

	g3 = gravity(p3, alpha=1, beta=1, gamma=gamma)

	# Use definition of m_b with correction for the intra-location flow
	if tildeM:
		sizeb = tilde_m(size, g3)
	# use traditional definition of population mass m_b
	else:
		sizeb = 2*size 

	# Insert locations into two-point
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, sizeb]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)

	g2 = gravity(p2, 1, 1, gamma)

	eps = epsilon(g2, g3, ib=ib)

	return eps


def epsilon_r(x,y,N, size=1., ib=True, seed=False, tildeM=False):
	'''
	Epsilon for radiation model

	'''
	p2, p3, loci, locb = createPops(x,y,N,size,seed)

	r3 = radiation(p3)

	# Use definition of m_b with correction for the intra-location flow
	if tildeM:
		sizeb = tilde_m(size, r3)
	# use traditional definition of population mass m_b
	else:
		sizeb = 2*size 

	# Insert locations into two-point
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, sizeb]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)

	r2 = radiation(p2)

	eps = epsilon(r2, r3, ib=ib)

	return eps


def epsilon_io(x,y,N, size=1., ib=True, seed=False, tildeM=False, gamma=1.):
	'''
	Epsilon for intervening opportunities model

	'''
	p2, p3, loci, locb = createPops(x,y,N,size,seed)

	io3 = opportunities(p3, gamma)

	# Use definition of m_b with correction for the intra-location flow
	if tildeM:
		sizeb = tilde_m(size, io3)
	# use traditional definition of population mass m_b
	else:
		sizeb = 2*size 

	# Insert locations into two-point
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, sizeb]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)

	io2 = opportunities(p2, gamma)

	eps = epsilon(io2, io3, ib=ib)

	return eps


def anaTP(xmin, xmax, ymin, ymax, n, N, model='gravity', ib=True, heatmap=True, tildeM=False, func=gravity):
	'''
	Finds values of epsilon in an nxn 2D sample space for x and y at fixed N random locations and plots a heatmap
	ib is a boolean to consider the direction of flow (see docstring for epsilon)

	'''

	x = np.linspace(xmin, xmax, n)
	y = np.linspace(ymin, ymax, n)

	xy = np.array(np.meshgrid(x, y)).T # 2D Sample Space
	xy = np.swapaxes(xy,0,1)

	epsVals = np.zeros((n,n))

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	if model=='gravity':
		func = epsilon_g
	if model=='radiation':
		func = epsilon_r
	if model=='opportunities':
		func = epsilon_io

	for j, row in enumerate(xy): # fill sample space
		for i, pair in enumerate(row):
			epsVals[j][i] = abs(func(pair[0], pair[1], N, ib=ib, seed=seed, tildeM=tildeM))

	nanMask = np.isnan(epsVals)

	epsVals[nanMask] = 1

	epsVals = np.flip(epsVals, 0) # make the y axis ascend

	xticks = np.around(x*np.sqrt(N), 2)
	yticks = np.flip(np.around(y*np.sqrt(N), 2), 0) # make the y axis ascend

	if heatmap is True: # Plot heatmap for eps with x-y
		ax = sns.heatmap(epsVals, xticklabels=xticks, yticklabels=yticks, square=True)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		ax.set_xlabel(r'$x \sqrt{N}$')
		ax.set_ylabel(r'$y \sqrt{N}$')

	else: # Plot contour
		plt.figure()
		ax = plt.contour(x*np.sqrt(N), np.flip(y*np.sqrt(N), 0), epsVals)
		plt.clabel(ax, inline=1, fontsize=10)

		plt.rc('text', usetex=True)
		# plt.rc('font', family='serif')

		plt.xlabel(r'$x \sqrt{N}$')
		plt.ylabel(r'$y \sqrt{N}$')

	# Plot location distribution
	# plt.figure()
	# plot_pop(pop_random(N, seed=seed))

	plt.show()

	return

def epsChangeY(ymin, ymax, x, n, N, ib=True):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations

	'''
	y = np.linspace(ymin, ymax, n)

	epsVals = []

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	for val in y:
		epsVals.append(abs(epsilon(x, val, N, ib=ib, seed=seed)))

	yEps = np.array([y * np.sqrt(N), np.array(epsVals)]).T

	ax = sns.regplot(yEps[:,0], yEps[:,1], scatter_kws={'s':10}, fit_reg=False)

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	ax.set_xlabel(r'$y \sqrt{N}$')
	ax.set_ylabel(r'$\epsilon$')

	plt.show()

	return





