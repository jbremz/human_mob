# Contains functions to justify the assumption that k_ib \approx k_ij (or k_ik) 

from hm.hm_models.gravity import gravity
from hm.pop_models.pop_random import random as pop_random
from hm.analysis.explicit_tripoint import *
import numpy as np
from matplotlib import pyplot as plt

def k_ratio(r_ib, r_jk, N, gamma=2, exp=True):
	'''
	Returns the ratio of K values for tri and two-point distributions with certain parameter values 

	'''
	size = 1
	tildeM=2

	p2, p3, loci, locb = createPops(r_ib,r_jk,N,size,seed=False)

	g3 = gravity(p3, alpha=1, beta=1, gamma=gamma, exp=exp)

	# Insert locations into two-point
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, tildeM]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)

	g2 = gravity(p2, 1, 1, gamma, exp=exp)

	k3 = g3.K(0)
	k2 = g2.K(0)

	return k2/k3

def ratio_N(r_ib, r_jk, Nmin, Nmax, n, runs=1, size=1, gamma=2, exp=True, tildeM=2):
	'''
	Plots ratio of the K values for varying number of locations N

	'''

	N = np.linspace(Nmin, Nmax, n, dtype=int)

	ratios = []
	sRatios = []

	for val in N:
		tempRatios = []
		for run in range(runs):
			tempRatios.append(k_ratio(r_ib, r_jk, val, gamma, exp))
		ratios.append(np.mean(np.array(tempRatios)))
		sRatios.append(np.std(np.array(tempRatios)))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.errorbar(N, ratios, yerr=sRatios, elinewidth=1, fmt='o', ms=4)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$N$')
	ax.set_ylabel(r'$\frac{K_{ij}}{K_{ib}}$')

	plt.title('Normalisation constant ratio for gravity model - ' + r'$r_{ib} = %s$' % r_ib + ', ' + r'$r_{jk} = %s$' % r_jk + ', ' + r'$\gamma = %s$' % gamma)

	plt.show()

	return

def ratio_rjk(r_ib, r_jkMin, r_jkMax, n, N, runs=1, size=1, gamma=2, exp=True):
	'''
	Plots ratio of the K values for varying number of locations N

	'''

	r_jk = np.linspace(r_jkMin, r_jkMax, n)

	ratios = []
	sRatios = []

	for val in r_jk:
		tempRatios = []
		for run in range(runs):
			tempRatios.append(k_ratio(r_ib, val, N, gamma, exp))
		ratios.append(np.mean(np.array(tempRatios)))
		sRatios.append(np.std(np.array(tempRatios)))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

	ax.errorbar(r_jk, ratios, yerr=sRatios, elinewidth=1, fmt='o', ms=4)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{jk}$')
	ax.set_ylabel(r'$\frac{K_{ij}}{K_{ib}}$')

	plt.title('Normalisation constant ratio for gravity model - ' + r'$r_{ib} = %s$' % r_ib + ', ' + r'$N = %s$' % N + ', ' + r'$\gamma = %s$' % gamma)

	plt.show()

	return




