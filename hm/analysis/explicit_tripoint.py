# Explicit tri-point model
# Place three locations at explicitly chosen positions amongst a random population distribution and study the fluxes between them in different conditions


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
from scipy.stats import chisquare
from tqdm import tqdm
from hm.utils.utils import time_label

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-deep')

# fig = plt.figure(figsize=(1200/110.27, 1000/110.27), dpi=110.27)
# ax.legend(frameon=False)

# ------------------ ERROR FUNCTIONS ------------------

def createPops(x,y,N,size,seed):
	'''
	Returns the tripoint and two-point population objects as well as location of i and location of b

	'''
	if type(seed) is bool:
		p = pop_random(N)

	else: # use seed for random population distribution so that it can be replicated
		p = pop_random(N, seed=seed)

	# Changed these locations so that j an k remain at constant x - less variation in population distribution
	loci = [0.5+x, 0.5]
	locj = [0.5, 0.5-y/2.]
	lock = [0.5, 0.5+y/2.]
	locb = [0.5, 0.5]

	p3 = copy.deepcopy(p) # the tripoint arrangement
	p2 = copy.deepcopy(p) # the two-point arrangement

	p3.locCoords = np.insert(p3.locCoords, 0, np.array([loci, locj, lock]), axis=0) # TODO - careful, there will be > N locations here now
	p3.popDist = np.insert(p3.popDist, 0, np.array([size, size, size]), axis=0)

	# re-initialise parameters
	p3.DM, p3.size = p3.distance_matrix(), len(p3.locCoords)

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
		eps = 1 - (mobObj3.flux(0,1, probs=False)+mobObj3.flux(0,2))/mobObj2.flux(0,1, probs=False)

	else: # from the satellites to i 
		mob2Flux = mobObj2.flux(1,0)
		eps = (mob2Flux - (mobObj3.flux(1,0, probs=False)+mobObj3.flux(2,0,probs=False)))/mob2Flux

	return eps

def epsilon_g(x,y,N, size=1., ib=True, exp=True, seed=False, tildeM=2, gamma=20):

	'''
	Takes the x and y displacements defined in the tripoint problem and returns
	the error between treating the satellite locations as one and as separate

	seed = False --> use completely random pop dist
	seed = int --> recreate a previous population distribution

	'''
	p2, p3, loci, locb = createPops(x,y,N,size,seed)

	g3 = gravity(p3, alpha=1, beta=1, gamma=gamma, exp=exp)

	# Insert locations into two-point
	p2.popDist = np.insert(p2.popDist, 0, np.array([size, tildeM]), axis=0)
	p2.locCoords = np.insert(p2.locCoords, 0, np.array([loci, locb]), axis=0)

	# re-initialise parameters
	p2.DM, p2.size = p2.distance_matrix(), len(p2.locCoords)

	g2 = gravity(p2, 1, 1, gamma, exp=exp)

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

	# re-initialise parameters
	p2.DM, p2.size = p2.distance_matrix(), len(p2.locCoords)

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

def anlyt_k(rmin, rmax, gamma, rho, exp=True):
	'''
	K approximation in a disc with a hole in it

	'''

	if exp:
		k = (2*np.pi * rho * (np.exp(-gamma*rmin)*(gamma*rmin+1)-np.exp(-gamma*rmax)*(gamma*rmax+1))/(gamma**2))**(-1)

	return k

def anlyt_epsilon_g(r_ib,r_jk,gamma, N=1000, exp=True, tildeM=False):
	'''	
	Returns the analytical result for epsilon

	TODO: include tilde m correction for power law
	TODO: tilde m only works for uniform population distribution
	TODO: include the population mass prefactor for non uniform population masses.

	'''
	r_ij = np.sqrt(r_ib**2 + (r_jk/2)**2)

	if exp:
		if type(tildeM) != bool:
			eps = 1 - (2*np.exp(-gamma*(r_ij)))/(tildeM*np.exp(-gamma*(r_ib)))
		else:
			eps = 1 - np.exp(-gamma * (r_ij - r_ib))

	else:
		eps = 1 - (r_ij/r_ib)**(-gamma)

	return eps

def anlyt_epsilon_r(r_ib,r_jk, N=1000):
	'''	
	Returns the analytical result for epsilon

	TODO: include tilde m correction for power law
	TODO: tilde m only works for uniform population distribution
	TODO: include the population mass prefactor for non uniform population masses.

	'''
	r_ij = np.sqrt(r_ib**2 + (r_jk/2)**2)
	rho = N

	eps = 1 - (r_ib**2/r_ij**2)*(np.pi * rho * r_ib**2 + 2)/(np.pi * rho * r_ij**2 + 1)

	return eps


# ------------------ ANALYSIS FUNCTIONS ------------------

def gridEpsilon(xy, N, ib, tildeM, func):
	'''
	Returns array of filled with epsilon values for the given x and y values and epsilon function

	'''
	epsVals = np.zeros((len(xy),len(xy), 1)) # create empty array for epsilon

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distributions are the same

	for j, row in enumerate(xy): # fill sample space
		for i, pair in enumerate(row):
			epsVals[j][i][0] = abs(func(pair[0], pair[1], N, ib=ib, seed=seed, tildeM=tildeM)) # TODO - not very efficient, would be better just to keep the distribution and change the single points

	nanMask = np.isnan(epsVals)

	epsVals[nanMask] = 0 # TODO - this could cause interpretation issues

	return epsVals, seed

def anaTP(xmin, xmax, ymin, ymax, n, N, runs=1, model='gravity', ib=True, heatmap=True, tildeM=False):
	'''
	Finds values of epsilon in an nxn 2D sample space for x and y at fixed N random locations and plots a heatmap
	ib is a boolean to consider the direction of flow (see docstring for epsilon)

	'''

	x = np.linspace(xmin, xmax, n)
	y = np.linspace(ymin, ymax, n)

	xy = np.array(np.meshgrid(x, y)).T # 2D Sample Space
	xy = np.swapaxes(xy,0,1)

	# Choose the correct epsilon function
	if model=='gravity':
		func = epsilon_g
	if model=='radiation':
		func = epsilon_r
	if model=='opportunities':
		func = epsilon_io

	epsVals, seed = gridEpsilon(xy, N, ib, tildeM, func) # returns the epsilon values in the grid

	for i in np.arange(runs-1):
		epsVals = np.concatenate((epsVals,gridEpsilon(xy, N, ib, tildeM, func)[0]), axis=2) 

	meanEps = np.mean(epsVals, axis=2)
	sigmaEps = np.std(epsVals, axis=2)/np.sqrt(runs) # TODO - is this the right treatment of error?

	# Plotting

	meanEps = np.flip(meanEps, 0) # make the y axis ascend

	xticks = np.around(x*np.sqrt(N), 2)
	yticks = np.flip(np.around(y*np.sqrt(N), 2), 0) # make the y axis ascend

	if heatmap is True: # Plot heatmap for eps with x-y
		plt.figure()
		ax = sns.heatmap(meanEps, xticklabels=xticks, yticklabels=yticks, square=True)

		plt.rc('text', usetex=True)
		# plt.rc('font', family='serif')

		ax.set_xlabel(r'$r_{ib} \sqrt{N}$')
		ax.set_ylabel(r'$r_{jk} \sqrt{N}$')

		ax.set_title(model.title() + ' model ' + r'$\epsilon(r_{ib},r_{jk})$' + ' - ' + str(N) + ' locations, ' + str(runs) + ' run(s)')

		if runs > 1: # sigma epsilon plot
			plt.figure()
			ax2 = sns.heatmap(sigmaEps, xticklabels=xticks, yticklabels=yticks, square=True)

			ax2.set_xlabel(r'$r_{ib} \sqrt{N}$')
			ax2.set_ylabel(r'$r_{jk} \sqrt{N}$')
			ax2.set_title(model.title() + ' model ' + r'$\sigma_{\epsilon}(r_{ib},r_{jk})$' + ' - ' + str(N) + ' locations, ' + str(runs) + ' run(s)')

	else: # Plot contour
		plt.figure()
		ax = plt.contour(x*np.sqrt(N), np.flip(y*np.sqrt(N), 0), meanEps)
		plt.clabel(ax, inline=1, fontsize=10)

		plt.rc('text', usetex=True)
		# plt.rc('font', family='serif')

		plt.xlabel(r'$r_{ib} \sqrt{N}$')
		plt.ylabel(r'$r_{jk} \sqrt{N}$')

		plt.title(model.title() + ' model ' + r'$\epsilon(r_{ib},r_{jk})$' + ' - ' + str(N) + ' locations, ' + str(runs) + ' run(s)')

		if runs > 1: # sigma epsilon plot
			plt.figure()
			ax2 = plt.contour(x*np.sqrt(N), np.flip(y*np.sqrt(N), 0), sigmaEps)
			plt.clabel(ax2, inline=1, fontsize=10)

			plt.xlabel(r'$r_{ib} \sqrt{N}$')
			plt.ylabel(r'$r_{jk} \sqrt{N}$')

			plt.title(model.title() + ' model ' + r'$\sigma_{\epsilon}(r_{ib},r_{jk})$' + ' - ' + str(N) + ' locations, ' + str(runs) + ' run(s)')

	if runs == 1:
		plotLocs(N, seed, xmin, xmax, ymin, ymax, show=False)

	plt.show()

	# return sigmaEps

	return

def plotLocs(N, seed, xmin, xmax, ymin, ymax, show=True):
	'''
	Plots the spatial population distribution and superimposed lines showing the variation in x and y

	'''

	plt.figure()

	# Plot all the locations
	plot_pop(pop_random(N, seed=seed), show=False)

	# Plot the changes in r_ib and r_jk
	plt.plot([0.5 + xmin, 0.5 + xmax],[0.5, 0.5], 'orange', lw=1, label='i')
	plt.plot([0.5, 0.5],[0.5-0.5*ymax, 0.5-0.5*ymin], 'violet', lw=1, label='j')
	plt.plot([0.5, 0.5],[0.5+0.5*ymax, 0.5+0.5*ymin], 'mediumseagreen', lw=1, label='k')

	plt.legend(loc='upper right', shadow=True)

	if show:
		plt.show()

	return

def epsChangeY(ymin, ymax, x, n, N, ib=False, analytical=False, gamma=2, exp=True, tildeM=2):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations for gravity model.

	'''
	y = np.linspace(ymin, ymax, n)
	epsVals = []
	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	for val in tqdm(y):
		epsVals.append(abs(epsilon_g(x, val, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM)))

	yEps = np.array([y * np.sqrt(N), np.array(epsVals)]).T

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	ax.scatter(yEps[:,0], yEps[:,1], s=20, label='Simulation', color='C5', marker='x')

	if analytical:
		anlytYEps = np.array([y * np.sqrt(N), anlyt_epsilon_g(x, y, gamma=gamma, N=N, exp=exp, tildeM=tildeM)]).T
		ax.plot(anlytYEps[:,0], anlytYEps[:,1], label='Analytical', color='grey')

	ax.legend(frameon=False, fontsize=20)

	# plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=20)
	ax.set_ylabel(r'$\epsilon$', fontsize=20)
	plt.tick_params(axis='both', labelsize=15)
	ax.ticklabel_format(style='sci')

	plt.ylim(-0.00005,0.0005)
	plt.tight_layout()

	plt.title(r'$r_{ib}=0.4, N=$'+str(N))

	plt.savefig(time_label())

	# plt.show()

	return

def epsChangeY_r(ymin, ymax, x, n, N, runs=1, ib=False, analytical=False):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations for gravity model.

	'''
	y = np.linspace(ymin, ymax, n)
	epsVals = np.zeros((n, 1))
	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	# First run
	for i, val in enumerate(y):
		epsVals[i] = epsilon_r(x, val, N, ib=ib, seed=seed)

	# rest of the runs
	for i in tqdm(range(runs-1)):
		tempVals = np.zeros((n, 1))
		seed = int(np.random.rand(1)[0] * 10000000)
		for j, val in enumerate(y):
			tempVals[j] = epsilon_r(x, val, N, ib=ib, seed=seed)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	yEps = np.array([y * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	ax.scatter(yEps[:,0], yEps[:,1], s=20, label='Simulation', color='C4', marker='x')

	if analytical:
		anlytYEps = np.array([y * np.sqrt(N), anlyt_epsilon_r(x, y, N=N)]).T
		ax.plot(anlytYEps[:,0], anlytYEps[:,1], label='Analytical', color='grey')

	ax.legend(frameon=False, fontsize=20)

	ax.errorbar(yEps[:,0], yEps[:,1], yerr=sigmaEps, elinewidth=1, fmt='o', ms=2, color='C4')

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=20)
	ax.set_ylabel(r'$\epsilon$', fontsize=20)
	plt.tick_params(axis='both', labelsize=15)
	ax.ticklabel_format(style='sci')

	plt.title(r'$r_{ib}=0.4, N=$'+str(N))

	plt.autoscale(enable=True)

	plt.tight_layout()

	plt.savefig(time_label())

	plt.clf()

	return

def epsChangeYRatio_g(ymin, ymax, x, n, N, runs=1, ib=False, gamma=2, exp=True, tildeM=False):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations for gravity model.

	'''
	y = np.linspace(ymin, ymax, n)

	epsVals = np.zeros((n,1))

	for i, val in enumerate(y):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		epsVals[i] = epsilon_g(x, val, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM)

	for i in np.arange(runs-1):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		tempVals = np.zeros((n,1))
		for l, val in enumerate(y):
			tempVals[l] = epsilon_g(x, val, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	yEps = np.array([y * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	anlytYEps = np.array([y * np.sqrt(N), anlyt_epsilon_g(x, y, N=N, gamma=gamma, exp=exp, tildeM=tildeM)]).T
	ax.scatter(yEps[:,0], yEps[:,1]/anlytYEps[:,1], s=10)

	ax.errorbar(yEps[:,0], yEps[:,1]/anlytYEps[:,1], yerr=sigmaEps/anlytYEps[:,1], elinewidth=1, fmt='o', ms=2)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=15)
	ax.set_ylabel(r'$\frac{\bar{\epsilon_{sim}}}{\epsilon_{ana}}$', fontsize=25)

	plt.autoscale(enable=True)

	plt.show()

	return

def epsChangeYRatio_r(ymin, ymax, x, n, N, runs=1, ib=True):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations for gravity model.

	'''
	y = np.linspace(ymin, ymax, n)
	epsVals = np.zeros((n, 1))
	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	# First run
	for i, val in enumerate(y):
		epsVals[i] = epsilon_r(x, val, N, ib=ib, seed=seed)

	# rest of the runs
	for i in range(runs-1):
		tempVals = np.zeros((n, 1))
		seed = int(np.random.rand(1)[0] * 10000000)
		for j, val in enumerate(y):
			tempVals[j] = epsilon_r(x, val, N, ib=ib, seed=seed)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	yEps = np.array([y * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	anlytYEps = np.array([y * np.sqrt(N), anlyt_epsilon_r(x, y, N=N)]).T
	ax.scatter(yEps[:,0], yEps[:,1]/anlytYEps[:,1], s=10)

	ax.errorbar(yEps[:,0], yEps[:,1]/anlytYEps[:,1], yerr=sigmaEps/anlytYEps[:,1], elinewidth=1, fmt='o', ms=2)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=15)
	ax.set_ylabel(r'$\frac{\bar{\epsilon_{sim}}}{\epsilon_{ana}}$', fontsize=25)

	plt.autoscale(enable=True)

	plt.show()

	return

def epsChangeX(xmin, xmax, y, n, N, ib=False, analytical=False, gamma=2, exp=True, tildeM=2):
	'''
	Fixes y and varies x across n values between ymin and ymax for a random distribution of N locations

	'''
	x = np.linspace(xmin, xmax, n)

	epsVals = []

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	for val in tqdm(x):
		epsVals.append(epsilon_g(val, y, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM))

	xEps = np.array([x * np.sqrt(N), np.array(epsVals)]).T

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	ax.scatter(xEps[:,0], xEps[:,1], s=20, label='Simulation', color='C5', marker='x')

	if analytical:
		anlytXEps = np.array([x * np.sqrt(N), anlyt_epsilon_g(x, y, N=N, gamma=gamma, exp=exp, tildeM=tildeM)]).T
		ax.plot(anlytXEps[:,0], anlytXEps[:,1], label='Analytical', color='grey')

	ax.legend(frameon=False, fontsize=20)

	ax.set_xlabel(r'$r_{ib} \sqrt{N}$', fontsize=20)
	ax.set_ylabel(r'$\epsilon$', fontsize=20)
	plt.tick_params(axis='both', labelsize=15)
	ax.ticklabel_format(style='sci')

	plt.title(r'$r_{jk}=0.03, N=100$')

	plt.tight_layout()

	plt.savefig(time_label())

	plt.clf()

	return

def epsChangeX_r(xmin, xmax, y, n, N, runs=1, ib=False, analytical=False):
	'''
	Fixes y and varies x across n values between ymin and ymax for a random distribution of N locations

	'''
	x = np.linspace(xmin, xmax, n)

	epsVals = np.zeros((n,1))

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	# First run
	for i, val in enumerate(x):
		epsVals[i] = epsilon_r(val, y, N, ib=ib, seed=seed)

	# rest of the runs
	for i in range(runs-1):
		tempVals = np.zeros((n, 1))
		seed = int(np.random.rand(1)[0] * 10000000)
		for j, val in enumerate(x):
			tempVals[j] = epsilon_r(val, y, N, ib=ib, seed=seed)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	xEps = np.array([x * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	ax.scatter(xEps[:,0], xEps[:,1], s=10, label='Simulation')

	if analytical:
		anlytXEps = np.array([x * np.sqrt(N), anlyt_epsilon_r(x, y, N=N)]).T
		ax.scatter(anlytXEps[:,0], anlytXEps[:,1], s=10, label='Analytical Result')

	ax.errorbar(xEps[:,0], xEps[:,1], yerr=sigmaEps, elinewidth=1, fmt='o', ms=2)

	ax.legend(frameon=False)

	ax.set_xlabel(r'$r_{ib} \sqrt{N}$', fontsize=15)
	ax.set_ylabel(r'$\epsilon$', fontsize=15)

	plt.title(r'$r_{jk}=0.03, N=$' + str(N))

	plt.show()

	return

def epsChangeXRatio_g(xmin, xmax, y, n, N, runs=1, ib=False, gamma=2, exp=True, tildeM=2):
	'''
	Fixes y and varies x across n values between ymin and ymax for a random distribution of N locations

	'''
	x = np.linspace(xmin, xmax, n)
	epsVals = np.zeros((n,1))

	for i, val in enumerate(x):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		epsVals[i] = epsilon_g(val, y, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM)

	for i in np.arange(runs-1):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		tempVals = np.zeros((n,1))
		for l, val in enumerate(x):
			tempVals[l] = epsilon_g(val, y, N, ib=ib, seed=seed, gamma=gamma, exp=exp, tildeM=tildeM)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	xEps = np.array([x * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	anlytXEps = np.array([x * np.sqrt(N), anlyt_epsilon_g(x, y, N=N, gamma=gamma, exp=exp, tildeM=tildeM)]).T
	# ax.scatter(anlytXEps[:,0], xEps[:,1]/anlytXEps[:,1], s=10)

	ax.errorbar(xEps[:,0], xEps[:,1]/anlytXEps[:,1], yerr=sigmaEps/anlytXEps[:,1], elinewidth=1, fmt='o', ms=2)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{ib} \sqrt{N}$', fontsize=15)
	ax.set_ylabel(r'$\frac{\bar{\epsilon_{sim}}}{\epsilon_{ana}}$', fontsize=25)

	plt.show()

	return

def epsChangeXRatio_r(xmin, xmax, y, n, N, runs=1, ib=False):
	'''
	Fixes y and varies x across n values between ymin and ymax for a random distribution of N locations

	'''
	x = np.linspace(xmin, xmax, n)
	epsVals = np.zeros((n,1))

	for i, val in enumerate(x):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		epsVals[i] = epsilon_r(val, y, N, ib=ib, seed=seed)

	for i in np.arange(runs-1):
		seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same
		tempVals = np.zeros((n,1))
		for l, val in enumerate(x):
			tempVals[l] = epsilon_r(val, y, N, ib=ib, seed=seed)
		epsVals = np.concatenate((epsVals, tempVals), axis=1)

	meanEps = np.mean(epsVals, axis=1)
	sigmaEps = np.std(epsVals, axis=1)/np.sqrt(runs) # TODO - is this the right treatment of error?

	xEps = np.array([x * np.sqrt(N), np.array(meanEps)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	anlytXEps = np.array([x * np.sqrt(N), anlyt_epsilon_r(x, y, N=N)]).T
	ax.scatter(anlytXEps[:,0], xEps[:,1]/anlytXEps[:,1], s=10, label='Analytical Result')

	ax.errorbar(xEps[:,0], xEps[:,1]/anlytXEps[:,1], yerr=sigmaEps/anlytXEps[:,1], elinewidth=1, fmt='o', ms=2)

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{ib} \sqrt{N}$', fontsize=15)
	ax.set_ylabel(r'$\frac{\bar{\epsilon_{sim}}}{\epsilon_{ana}}$', fontsize=25)

	plt.show()

	return

def epsChangeGamma(gmin, gmax, r_ib, r_jk, n, N, ib=False, analytical=False):
	
	'''
	Fixes r_ib and r_jk and varies the gamma factor in the gravity model to produce different epsilon values

	'''
	gamma = np.linspace(gmin, gmax, n)

	epsVals = []

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	for val in gamma:
		epsVals.append(abs(epsilon_g(r_ib, r_jk, N, ib=ib, seed=seed, gamma=val, exp=exp)))

	gEps = np.array([gamma, np.array(epsVals)]).T

	fig = plt.figure(figsize=(1500/110.27, 1200/110.27), dpi=110.27)
	ax = fig.add_subplot(111)

	ax.scatter(gEps[:,0], gEps[:,1], s=10, label='Simulation')

	if analytical:
		anlytGEps = np.array([gamma, anlyt_epsilon_g(r_ib, r_jk, gamma=gamma, N=N)]).T
		ax.scatter(anlytGEps[:,0], anlytGEps[:,1], s=10, label='Analytical Result')

	ax.legend()

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$\gamma$')
	ax.set_ylabel(r'$\epsilon$')

	plt.title('Gravity model ' + r'$\epsilon(\gamma)$' + ' - ' + str(N) + ' locations, ' + '$r_{ib}$ = ' + str(r_ib) + ', $r_{jk}$ = ' + str(r_jk))

	plt.show()

	return








