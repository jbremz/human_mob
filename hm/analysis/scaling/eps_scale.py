import numpy as np 
from hm.coarse_grain.clustering import Clusters
from hm.pop_models.pop_explicit import explicit as pop_explicit
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.coarse_grain.coarse_matrix import epsilon_matrix
from hm.coarse_grain.coarse_matrix import coarse_grain as coarse_grain_matrix
from hm.analysis.scaling.pop_tools import make_pop

def iterate(df, d_max):
	"""Returns a list of Clusters objects with all the levels up to specified level."""

	# Clustering starts at level 1 (level 0 is unclustered)
	if len(d_max) < 1:
		print("iterate() only accepts clustering levels > 0")
		return

	pop = make_pop(df)
	levels = []

	for i in d_max:
		clusters = Clusters(pop, i)
		levels.append(clusters)
		
	return levels		
	
def cluster_population(cluster, pw = True):
	"""
	Returns an explicit population distribution.
	
	Takes a Clusters object and returns an explicit (pop_distribution) object 
	which has the PW-centroid of the cluster as location coordinates and the 
	total population of the cluster as the location population.
	""" 
	if not isinstance(cluster, Clusters):
		raise NameError("cluster must be a Clusters object")
		
	m = cluster.clustered_pop
	if pw == True:
		xy = cluster.pw_centroids().T
	else:
		xy = cluster.centroids().T
	pop = pop_explicit(xy, m)
	
	return pop

def gravity_ODM(clusters_list, level, gamma):
	"""
	Returns the ODM for the gravity model at a specific level of clustering.
	
	If level = 0 (no clustering), a dataframe needs to be specified so that 
	a population object can be created from it.
	"""
	
	pop = clusters_list[0].pop
	
	if level > 0:
		clustering = clusters_list[level-1]
		pop = cluster_population(clustering)

	g = gravity(pop, 1, 1, gamma)
	return g.ODM()

def reduced_ODM(clusters_list, level, gamma, df):
	"""Returns ODM for the combined flow between locations."""
	if level != 0:
		original_ODM = gravity_ODM(clusters_list, 0, gamma, df)
		clust = clusters_list[level-1].clusters
		reduced_ODM = coarse_grain_matrix(original_ODM, clust)
	else: # not reduced
		reduced_ODM = gravity_ODM(clusters_list, level, gamma, df)
	return reduced_ODM

def epsilon(clusters_list, level, gamma, df):
	"""Returns the epsilon matrix (defined compared to the ODM at the original location resolution) at a specific level of clustering."""
	clustered_ODM = gravity_ODM(clusters_list, level, gamma, df)
	combined_ODM = reduced_ODM(clusters_list, level, gamma, df)
	epsilon = epsilon_matrix(combined_ODM, clustered_ODM)
	return epsilon
