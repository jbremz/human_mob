# Contains functions to justify the assumption that k_ib \approx k_ij (or k_ik) 

from hm.hm_models.gravity import gravity
from hm.pop_models.pop_random import random as pop_random
from hm.analysis.explicit_tripoint import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from hm.utils.utils import time_label

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-deep')

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

	k3 = g3.K[0]
	k2 = g2.K[0]

	return k2/k3

def ratio_N(r_ib, r_jk, Nmin, Nmax, n, runs=1, size=1, gamma=2, exp=True, tildeM=2):
	'''
	Plots ratio of the K values for varying number of locations N

	'''

	N = np.linspace(Nmin, Nmax, n, dtype=int)

	ratios = []
	sRatios = []

	for val in tqdm(N):
		tempRatios = []

		# Lenormand formula
		S = 1/val # mean unit area
		if exp:
			gamma = 0.3 * (S)**(-0.18)
		else:
			gamma = 1.4 * (S)**(0.11)

		r_jk = 1/np.sqrt(val)

		for run in range(runs):
			tempRatios.append(k_ratio(r_ib, r_jk, val, gamma, exp))
		ratios.append(np.mean(np.array(tempRatios)))
		sRatios.append(np.std(np.array(tempRatios))/np.sqrt(runs))

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	ax.errorbar(N, ratios, yerr=sRatios, elinewidth=1, fmt='x', ms=4, color='C5')

	ax.legend(frameon=False)

	ax.set_xlabel(r'$N$', fontsize=20)
	ax.set_ylabel(r'$\frac{K_{ij}}{K_{ib}}$', fontsize=20)
	plt.tick_params(axis='both', labelsize=15)

	plt.title('Normalisation constant ratio for gravity model - ' + r'$r_{ib} = %s$' % r_ib)

	plt.tight_layout()

	plt.savefig(time_label())

	return

def ratio_rjk(r_ib, r_jkMin, r_jkMax, n, N, runs=1, size=1, gamma=None, exp=True):
	'''
	Plots ratio of the K values for varying r_jk

	'''
	S = 1/N # mean unit area
	if exp:
		gamma = 0.3 * (S)**(-0.18)
	else:
		gamma = 1.4 * (S)**(0.11)

	r_jk = np.linspace(r_jkMin, r_jkMax, n)

	ratios = []
	sRatios = []

	for val in tqdm(r_jk):
		tempRatios = []
		for run in range(runs):
			tempRatios.append(k_ratio(r_ib, val, N, gamma, exp))
		ratios.append(np.mean(np.array(tempRatios)))
		sRatios.append(np.std(np.array(tempRatios)))

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

	ax.errorbar(r_jk * np.sqrt(N), ratios, yerr=sRatios, elinewidth=1, fmt='x', ms=4, color='C5')

	ax.legend(frameon=False)

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=20)
	ax.set_ylabel(r'$\frac{K_{ij}}{K_{ib}}$', fontsize=20)
	plt.tick_params(axis='both', labelsize=15)

	plt.ylim(0.8, 1.3)

	plt.title('Normalisation constant ratio for gravity model - ' + r'$r_{ib} = %s$' % r_ib + ', ' + r'$N = %s$' % N)

	plt.tight_layout()

	plt.savefig(time_label())

	return




