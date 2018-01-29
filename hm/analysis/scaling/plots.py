import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from tqdm import tqdm_notebook as tqdm

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

def eps_hier(pop_hier_obj):
	'''
	Takes a pop_hier object

	Returns:

	- List of epsilon matrices at each level
	- List of distance matrices at each level

	To be used with eps_distance_hier()

	'''
	h = pop_hier_obj

	epsList = []
	DMList = []

	for level in tqdm(range(len(h.levels)+1)):
		epsList.append(h.epsilon(level))
		DMList.append(h.DM_level(level))

	return epsList, DMList

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
	bins = np.linspace(xMin, xMax, N)

	mean_epss = []
	sigma_epss = []

	# Find mean and std. for each distance bin (across all the clustering levels)
	for i in range(len(epsList)):
		inds = np.digitize(DMTris[i], bins) # indices of the distance bins to which each eps value belongs
		
		mean_eps = []
		sigma_eps = []

		for b in np.arange(len(bins)):
			mask = inds == b
			e = epsTris[i][mask]
			mean_eps.append(np.mean(e))
			sigma_eps.append(np.std(e)/np.sqrt(len(e)))

		mean_epss.append(mean_eps)
		sigma_epss.append(sigma_eps)

	# Alter figsize here
	fig = plt.figure(figsize=(11,8))
	ax = fig.add_subplot(111)

	for i in range(len(mean_epss)):
		ax.errorbar(bins, mean_epss[i], elinewidth=1, fmt='o', ms=4, yerr=sigma_epss[i], label=r'$d_{max} = $' + str(d_maxs[i]))

	# Axes labels & Title

	if ib:
		flow = 'forward-flow'
	else:
		flow = 'backwards-flow'

	ax.set_xlabel(r'Distance (m)', fontsize=15)
	ax.set_ylabel(r'$<\epsilon>$', fontsize=15)
	ax.set_title(r'Mean $\epsilon$ at different levels of clustering (' + flow + ')')
	ax.legend()

	return

