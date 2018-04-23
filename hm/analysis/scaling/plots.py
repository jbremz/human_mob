import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from tqdm import tqdm
import scipy as sp
from hm.utils.utils import time_label
import matplotlib.ticker as plticker

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-deep')

# fig = plt.figure(figsize=(1000/110.27, 800/110.27), dpi=110.27)
# ax.legend(frameon=False)
# plt.grid(linestyle='--', linewidth=0.5)
# fontsize=15

def eps_heatmap(M, model='gravity'):
	'''
	Takes epsilon matrix and plots a heatmap for the locations
	'''

	ax = sns.heatmap(M, square=True)
	plt.rc('text', usetex=True)
	ax.set_xlabel('Location Origin')
	ax.set_xlabel('Location Destination')

	ax.set_title(r'$\epsilon$' + ' in the ' + model + ' model')

	plt.show()

	return 

def eps_distance(eps, DM, N, ib = True, model='gravity'):
	'''
	Takes the epsilon matrix and ODM returns a histogram (with N bins) of the epsilon values against distance binned between all the cluster locations.
	
	'''
	# TODO could put this in a function to be neater
	if ib:
		# Only take the upper triangle
		iu = np.array(np.triu_indices(eps.shape[0]))
		diag = iu[0] == iu[1]
		iu = list(iu[:,~diag]) # take out the diagonal
		epsTri = eps[iu]
		DMTri = DM[iu]
	else:
		# Only take lower triangle (backflows)
		il = np.array(np.tril_indices(eps.shape[0]))
		diag = il[0] == il[1]
		il = list(il[:,~diag]) # take out the diagonal
		epsTri = eps[il]
		DMTri = DM[il]

	# bin the data
	xMin, xMax = np.min(DMTri), np.max(DMTri)
	bins = np.linspace(xMin, xMax, N)
	inds = np.digitize(DMTri, bins) # indices of the distance bins to which each eps value belongs

	mean_eps = []
	sigma_eps = []

	for b in np.arange(len(bins)):
		mask = inds == b
		e = epsTri[mask]
		mean_eps.append(np.mean(e))
		sigma_eps.append(np.std(e)/np.sqrt(len(e)))


	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.errorbar(bins, mean_eps, elinewidth=1, fmt='o', ms=4, yerr=sigma_eps)

	clusterNum = eps.shape[0]
	if ib:
		flow = 'forward-flow'
	else:
		flow = 'backwards-flow'

	ax.set_xlabel(r'Distance (m)', fontsize=15)
	ax.set_ylabel(r'$<\epsilon>$', fontsize=15)
	ax.set_title(r'Mean $\epsilon$ for ' + str(clusterNum) + ' clusters (' + flow + ')')

	plt.show()

	return

def eps_hier(pop_hier_obj, model='g', gamma=False, exp=False): 
	'''
	Takes a pop_hier object

	Returns:

	- List of epsilon matrices at each level

	To be used with eps_distance_hier()

	'''

	if model != 'g' and model != 'r':
		print("Please input 'g':gravity, 'r':radiation")
		return
		
	h = pop_hier_obj

	epsList = []

	for level in tqdm(range(1,len(h.levels)+1)): # only go from level 1
		epsList.append(h.epsilon(level, model=model, gamma=gamma, exp=exp))

	return epsList

def DM_list(pop_hier_obj):
	'''
	Takes a pop_hier object

	Returns:

		- List of distance matrices at each level
	'''
	h = pop_hier_obj

	DMList = []

	for level in tqdm(range(1,len(h.levels)+1)): # only go from level 1
		DMList.append(h.DM_level(level))

	return DMList
	
	
	
def eps_distance_hier(epsList, DMList, d_maxs, N, ib=True, model='gravity'):
	'''
	Plots epsilon against distances at different levels in the hierarchy

	'''
	epsTris = []
	DMTris = []

	if ib:
		for i in range(len(epsList)):
			# Only take the upper triangle
			iu = np.array(np.triu_indices(epsList[i].shape[0]))
			diag = iu[0] == iu[1]
			iu = list(iu[:,~diag]) # take out the diagonal
			epsTri = epsList[i][iu]
			DMTri = DMList[i][iu]
			epsTris.append(epsTri)
			DMTris.append(DMTri)
	else:
		for i in range(len(epsList)):
			# Only take lower triangle (backflows)
			il = np.array(np.tril_indices(eps.shape[0]))
			diag = il[0] == il[1]
			il = list(il[:,~diag]) # take out the diagonal
			epsTri = eps[i][il]
			DMTri = DM[i][il]
			epsTris.append(epsTri)
			DMTris.append(DMTris)

	# bins the data by distance
	xMin, xMax = np.min(DMTris[0]), np.max(DMTris[0]) # choose level 0 to define the bins (this will have the greatest extent in DM)
	bins = np.linspace(xMin, xMax-(xMax-xMin)/N, N) # the final bin edge is a bin-width before the maxmium distance value (so it has > 1 locations in it)

	mean_epss = []
	sigma_epss = []

	# Find mean and std. for each distance bin (across all the clustering levels)
	for i in range(len(epsList)):
		inds = np.digitize(DMTris[i], bins) # indices of the distance bins to which each eps value belongs
		
		mean_eps = []
		sigma_eps = []

		for b in np.arange(1,len(bins)+1):
			mask = inds == b
			e = epsTris[i][mask]
			mean_eps.append(np.mean(e))
			sigma_eps.append(np.std(e)/np.sqrt(len(e)))

		mean_epss.append(mean_eps)
		sigma_epss.append(sigma_eps)

	# Alter figsize here
	fig = plt.figure(figsize=(830/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	for i in range(len(mean_epss)):
		# labels = [0] + d_maxs # to include the base (no clustering) level
		labels = d_maxs
		colours = ['grey','C1','C2','C3','C4','C5']
		ax.errorbar(bins/1000, mean_epss[i], elinewidth=2, fmt='x', ms=8, yerr=sigma_epss[i], label=r'$d_{max} = $' + str(labels[i]) + 'm', mew=2, color=colours[i])
		# ax.errorbar(bins/1000, mean_epss[i], elinewidth=1, fmt='x', ms=6, yerr=sigma_epss[i], label=str(labels[i]))

	# Axes labels & Title

	if ib:
		flow = 'forward-flow'
	else:
		flow = 'backwards-flow'

	loc = plticker.MultipleLocator(base=0.2) # you can change the base as desired
	ax.yaxis.set_major_locator(loc)
	ax.set_xlabel(r'Distance (Km)', fontsize=30, labelpad=15)
	ax.set_ylabel(r'$\langle\epsilon \rangle$', fontsize=40)
	plt.ylim(0,1.3)
	ax.legend(frameon=False, fontsize=20, loc='upper right')
	ax.tick_params(labelsize=20)
	plt.tight_layout()

	plt.savefig(time_label())

	return

def gamma_S(hier, gamma_0, gamma_opts):
	'''
	Plots gamma against unit area given a pop_hier object and the optimised gammas

	'''
	# S = [np.mean(hier.pop.locArea)] # level 0
	S = []
	gammas = [gamma_0] + gamma_opts

	for level in hier.levels:
		S.append(np.mean(level.clustered_area)*10**(-6))

	x = np.log(S)
	# y = np.log(gammas)
	y = np.log(gamma_opts)

	# slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
	coeffs, cov = np.polyfit(x, y, 1, cov=True)

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	gam_fit = coeffs[0]*x + coeffs[1]

	ax.scatter(x,y)
	ax.plot(x, gam_fit, 'r', linewidth=0.5)

	ax.set_xlabel(r'$\log(<S>)$', fontsize=20)
	ax.set_ylabel(r'$\log(\gamma_{opt})$', fontsize=20)
	ax.set_title(r'exponent $=$' + str(coeffs[0]) + r'$\pm$' + str(np.sqrt(cov[0][0])))

	ax.legend(frameon=False, fontsize=15, loc='upper right')
	ax.tick_params(labelsize=15)
	plt.tight_layout()

	sigma_a = np.sqrt(cov[0][0])
	sigma_b = np.sqrt(cov[1][1])

	alpha = np.exp(coeffs[1])
	beta = coeffs[0]
	sigma_alpha = np.exp(coeffs[1])*sigma_b
	sigma_beta = sigma_a

	return alpha, beta, sigma_alpha, sigma_beta

def gamma_dmax(d_maxs, gamma_opts):
	'''
	Plots gamma against maximum cluster separation

	'''

	x = np.array(d_maxs) # add the value for no clustering i.e. d_max = 0 
	y = np.array(gamma_opts)

	# slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
	coeffs, cov = np.polyfit(x, y, 1, cov=True)

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	gam_fit = coeffs[0]*x + coeffs[1]

	ax.scatter(x,y)
	ax.plot(x, gam_fit, 'r', linewidth=0.5)

	ax.set_xlabel(r'$d_{max}$ (m)', fontsize=20)
	ax.set_ylabel(r'$\gamma_{opt}$', fontsize=20)
	ax.set_title(r'gradient $=$' + str(coeffs[0]) + r'$\pm$' + str(np.sqrt(cov[0][0])))

	ax.tick_params(labelsize=15)
	plt.tight_layout()

	return coeffs, np.sqrt(np.diag(cov))


def gamma_d(hier, gamma_opts):
	'''
	Plots gamma against average separation

	'''
	ds = [] # average separation 

	for level in range(1, len(hier.levels)+1):
		DM = hier.DM_level(level)
		iu = np.array(np.triu_indices(DM.shape[0]))
		diag = iu[0] == iu[1]
		iu = list(iu[:,~diag])
		DM = DM[iu]
		ds.append(np.mean(DM))

	x = np.array(ds)
	y = np.array(gamma_opts)

	# x = np.log(x)
	# y = np.log(y)

	coeffs, cov = np.polyfit(x, y, 1, cov=True)

	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	gam_fit = coeffs[0]*x + coeffs[1]

	ax.scatter(x/1000,y)
	ax.plot(x/1000, gam_fit, 'r', linewidth=0.5)

	ax.set_xlabel(r'Mean Location Separation (Km)', fontsize=20)
	ax.set_ylabel(r'$\gamma_{opt}$', fontsize=20)
	ax.set_title(r'gradient $=$' + str(coeffs[0]) + r'$\pm$' + str(np.sqrt(cov[0][0])))

	ax.tick_params(labelsize=15)
	plt.tight_layout()

	return coeffs, np.sqrt(np.diag(cov))

def eps_distance_compare(epsList, DMList, N, labels=[r'Lenormand $\gamma$', r'Optimised $\gamma$'], ib=True, model='gravity'):
	'''
	Plots epsilon against distances at the same scale for two different models/exponents etc.

	'''
	epsTris = []
	DMTris = []

	if ib:
		for i in range(len(epsList)):
			# Only take the upper triangle
			iu = np.array(np.triu_indices(epsList[i].shape[0]))
			diag = iu[0] == iu[1]
			iu = list(iu[:,~diag]) # take out the diagonal
			epsTri = epsList[i][iu]
			DMTri = DMList[i][iu]
			epsTris.append(epsTri)
			DMTris.append(DMTri)
	else:
		for i in range(len(epsList)):
			# Only take lower triangle (backflows)
			il = np.array(np.tril_indices(eps.shape[0]))
			diag = il[0] == il[1]
			il = list(il[:,~diag]) # take out the diagonal
			epsTri = eps[i][il]
			DMTri = DM[i][il]
			epsTris.append(epsTri)
			DMTris.append(DMTris)

	# bins the data by distance
	xMin, xMax = np.min(DMTris[0]), np.max(DMTris[0]) # choose level 0 to define the bins (this will have the greatest extent in DM)
	bins = np.linspace(xMin, xMax-(xMax-xMin)/N, N) # the final bin edge is a bin-width before the maxmium distance value (so it has > 1 locations in it)

	mean_epss = []
	sigma_epss = []

	# Find mean and std. for each distance bin (across all the clustering levels)
	for i in range(len(epsList)):
		inds = np.digitize(DMTris[i], bins) # indices of the distance bins to which each eps value belongs
		
		mean_eps = []
		sigma_eps = []

		for b in np.arange(1,len(bins)+1):
			mask = inds == b
			e = epsTris[i][mask]
			mean_eps.append(np.mean(e))
			sigma_eps.append(np.std(e)/np.sqrt(len(e)))

		mean_epss.append(mean_eps)
		sigma_epss.append(sigma_eps)

	# Alter figsize here
	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	for i in range(len(mean_epss)):
		# labels = [0] + d_maxs # to include the base (no clustering) level
		ax.errorbar(bins/1000, mean_epss[i], elinewidth=2, fmt='x', ms=6, mew=2, yerr=sigma_epss[i], label=labels[i])

	# Axes labels & Title

	if ib:
		flow = 'forward-flow'
	else:
		flow = 'backwards-flow'

	# loc = plticker.MultipleLocator(base=0.2) # you can change the base as desired
	# ax.yaxis.set_major_locator(loc)
	ax.set_xlabel(r'Distance (Km)', fontsize=25, labelpad=15)
	# ax.set_ylabel(r'$|\langle \epsilon \rangle|$', fontsize=30)
	ax.set_ylabel(r'$\langle \epsilon \rangle$', fontsize=30)
	# ax.set_title(r'Mean $\epsilon$ at different levels of clustering (' + flow + ')')
	ax.legend(frameon=False, fontsize=20, loc='upper right')
	ax.tick_params(labelsize=20)
	# plt.ylim(0,0.7)
	plt.tight_layout()

	plt.savefig(time_label())

	return

def n_clusters(hierList, labels=['London', 'Birmingham']):
	'''
	Plots a bar chart of the number of clusters for each level of clustering for a list of population hierarchy objects
	'''

	fig = plt.figure(figsize=(840/110.27, 800/110.27), dpi=300)
	ax = fig.add_subplot(111)

	d_maxs = list(map(str,['0'] + hierList[0].d_maxs))
	clusterNums = np.zeros((len(hierList),len(d_maxs)))
	index = np.arange(len(d_maxs))

	for i, hier in enumerate(hierList):

		clusterNums[i][0] = hier.pop.size

		for j, level in enumerate(hier.levels):
			clusterNums[i][j+1] = level.clusters_num

	bar_width = 0.35
	opacity = 0.7
	# opacity = 1
	colours = ['C1','C4']

	# for n, hier in enumerate(hierList):
	# 	rect = 'rect_{}'.format(n+1)
	# 	rect = ax.bar(index, clusterNums[n], bar_width, alpha=opacity, color=colours[n], label=labels[n])

	rect1 = ax.bar(index, clusterNums[0], bar_width, alpha=opacity, color=colours[0], label=labels[0])
	rect2 = ax.bar(index+bar_width, clusterNums[1], bar_width, alpha=opacity, color=colours[1], label=labels[1])

	ax.set_xlabel(r'$d_{max}$ (m)', fontsize=25, labelpad=15)
	ax.set_ylabel(r'$N_{clusters}$', fontsize=30, labelpad=15)
	ax.tick_params(labelsize=20)
	ax.set_xticks(index + bar_width / 2)
	ax.set_xticklabels(d_maxs)
	ax.legend(frameon=False, fontsize=20)

	plt.tight_layout()

	plt.savefig(time_label())

	return 



	


















