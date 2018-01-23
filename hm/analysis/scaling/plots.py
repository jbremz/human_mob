import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()


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
		# sigma_eps.append(np.std(e)/np.sqrt(len(e)))
		sigma_eps.append(np.std(e))

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

def eps_distance_hier(epsList, DM_list, N, ib=True, model='gravity'):
	'''
	

	'''



