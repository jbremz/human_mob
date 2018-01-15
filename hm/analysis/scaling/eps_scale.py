import numpy as np 
from hm.coarse_grain.clustering import Clusters
from hm.pop_models.pop_explicit import explicit as pop_explicit
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.coarse_grain.coarse_matrix import epsilon_matrix, reorder_matrix
from hm.coarse_grain.coarse_matrix import coarse_grain as coarse_grain_matrix
from matplotlib import pyplot as plt

def iterate(df, d_max, level = 1, pw = False):
	"""Returns a list of Clusters objects with all the levels up to specified level."""

	# Clustering starts at level 1 (level 0 is unclustered)
	if level < 1:
		print("iterate() only accepts clustering levels > 0")
		return

	x = np.array(df['Easting'])
	y = np.array(df['Northing'])
	m = np.array(df['TotPop2011'])
	xy = np.array([x, y]).T
	pop = pop_explicit(xy, m)
	clusters = Clusters(pop, d_max)
	levels = [clusters]

	for i in range(level-1):
		pop = cluster_population(clusters, pw)
		d_max = d_max + 2000 # TODO maybe change this to a scaling factor e.g. 2*d_max?
		clusters = Clusters(pop, d_max)
		levels.append(clusters)
		
	return levels

def viz_levels(df, cluster, pw):
	if not isinstance(cluster, Clusters):
		raise NameError("cluster must be a Clusters object")
	else:
		x = np.array(df['Easting'])
		y = np.array(df['Northing'])
		xy = np.array([x, y]).T
		if pw == True:
			plt.plot(cluster.pw_centroids()[0], cluster.pw_centroids()[1], '.')
		else:
			plt.plot(cluster.centroids()[0], cluster.centroids()[1], '.')
		plt.scatter(xy[:,0], xy[:,1], c = cluster.clusters)  
		plt.show()
		
	
def cluster_population(cluster, pw = False):
	"""
	Returns an explicit population distribution.
	
	Takes a Clusters object and returns an explicit (pop_distribution) object 
	which has the PW-centroid of the cluster as location coordinates and the 
	total population of the cluster as the location population.
	""" 
	if not isinstance(cluster, Clusters):
		raise NameError("cluster must be a Clusters object")
	else:
		m = cluster.clustered_pop
		if pw == True:
			xy = cluster.pw_centroids().T
		else:
			xy = cluster.centroids().T
		pop = pop_explicit(xy, m)
	return pop
	
def gravity_ODM(df, d_max, level, gamma = 0.2):
	"""Returns the ODM for the gravity model at a specific level of clustering."""
	if level == 0:
		x = np.array(df['Easting'])
		y = np.array(df['Northing'])
		m = np.array(df['TotPop2011'])
		xy = np.array([x, y]).T
		pop = pop_explicit(xy, m)

	else:
		clustering = iterate(df, d_max, level = level)[-1]
		pop = cluster_population(clustering)

	g = gravity(pop, 1, 1, gamma)
	return g.ODM()

def reordered_ODM(df, d_max, level, gamma = 0.2):
	"""Returns reordered ODM for the flow between clustered locations."""
	original_ODM = gravity_ODM(df, d_max, level-1, gamma)
	clust = iterate(df, d_max, level)[level-1].clusters
	reordered_ODM = reorder_matrix(original_ODM, clust)[0]
	return reordered_ODM

def reduced_ODM(df, d_max, level, gamma = 0.2):
	"""Returns ODM for the combined flow between locations."""
	if level != 0:
		original_ODM = gravity_ODM(df, d_max, level-1, gamma)
		clust = iterate(df, d_max, level)[level-1].clusters
		reduced_ODM = coarse_grain_matrix(original_ODM, clust)
	else: # not reduced
		reduced_ODM = gravity_ODM(df, d_max, level, gamma)
	return reduced_ODM

def multi_reduced_ODM(df, d_max, level, gamma = 0.2):
	"""Recursively combines flows at each clustering level and returns the final combined ODM"""

	# Original populations
	x = np.array(df['Easting'])
	y = np.array(df['Northing'])
	m = np.array(df['TotPop2011'])
	xy = np.array([x, y]).T
	pop = pop_explicit(xy, m)
	original_ODM = gravity(pop, 1, 1, gamma).ODM()

	ODM_init = original_ODM

	if level != 0:
		levels = iterate(df, d_max, level) # list of clusters at each level
		# Recursively reducing
		for i in range(level):
			ODM_init = coarse_grain_matrix(ODM_init, levels[i].clusters)

	return ODM_init

def epsilon(df, d_max, level, gamma = 0.2):

	"""Returns the epsilon matrix at a specific level of clustering."""
	clustered_ODM = gravity_ODM(df, d_max, level, gamma)
	combined_ODM = reduced_ODM(df, d_max, level, gamma)
	epsilon = epsilon_matrix(combined_ODM, clustered_ODM) # WARNING changed this round
	return epsilon

def epsilonB(df, d_max, level, gamma = 0.2):
	"""Returns the epsilon matrix (defined compared to the ODM at the original location resolution) at a specific level of clustering."""
	clustered_ODM = gravity_ODM(df, d_max, level, gamma)
	combined_ODM = multi_reduced_ODM(df, d_max, level, gamma)
	epsilon = epsilon_matrix(combined_ODM, clustered_ODM)
	return epsilon






