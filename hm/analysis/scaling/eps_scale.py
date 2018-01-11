import numpy as np 
import pandas
from hm.coarse_grain.clustering import Clusters
from hm.pop_models.pop_explicit import explicit as pop_explicit
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.coarse_grain.coarse_matrix import epsilon_matrix, reorder_matrix
from hm.coarse_grain.coarse_matrix import coarse_grain as coarse_grain_matrix

df = pandas.read_csv("/Users/Ilaria/Documents/Imperial/MSci_Project/Code/human_mob/data/CDRC/england_welsh_scotland_pwc2011.csv")
lon_df = df[df['Easting'] < 561966][df['Easting'] > 493024][df['Northing'] < 207218][df['Northing'] > 150912]

def iterate(df, d_max, times = 0):
	"""Returns a list of Clusters objects with all the levels up to level = times."""
	x = np.array(df)[:,5]
	y = np.array(df)[:,6]
	m = np.array(df)[:,2]
	xy = np.array([x, y])
	xy = np.transpose(xy)
	pop = pop_explicit(xy, m)
	clusters = Clusters(pop, d_max)
	levels = [clusters]
	
	if times > 0:
		for i in range(times):
			xy = np.transpose(clusters.centroids())
			m = clusters.clustered_pop
			pop = pop_explicit(xy, m)
			d_max = d_max + 2000
			clusters = Clusters(pop, d_max)
			levels.append(clusters)
	
	return levels

def gravity_ODM(df, d_max, level, gamma = 0.2):
	"""Returns the ODM for the gravity model at a specific level of clustering."""
	clustering = iterate(df, d_max, times = level)
	m = clustering[-1].clustered_pop
	xy = np.transpose(clustering[-1].centroids())
	pop = pop_explicit(xy, m)
	g = gravity(pop, 1, 1, gamma)
	return g.ODM()

def reordered_ODM(df, d_max, level, gamma = 0.2):
	"""Returns ODM for the flow between clustered locations."""
	original_ODM = gravity_ODM(df, d_max, level-1, gamma)
	clust = iterate(df, d_max, level)[level].clusters
	reordered_ODM = reorder_matrix(original_ODM, clust)[0]	
	return reordered_ODM

def reduced_ODM(df, d_max, level, gamma = 0.2):
	"""Returns ODM for the combined flow between locations."""
	if level == 0:
		x = np.array(df)[:,5]
		y = np.array(df)[:,6]
		m = np.array(df)[:,2]
		xy = np.array([x, y])
		xy = np.transpose(xy)
		pop = pop_explicit(xy, m)
		original_ODM = gravity(pop, 1, 1, gamma).ODM()
	else:
		original_ODM = gravity_ODM(df, d_max, level-1, gamma)
	clust = iterate(df, d_max, level)[level].clusters
	reduced_ODM = coarse_grain_matrix(original_ODM, clust)
	return reduced_ODM

def epsilon(df, d_max, level, gamma = 0.2):
	"""Retures the epsilon matrix at a specific level of clustering."""
	combined_ODM = reduced_ODM(df, d_max, level, gamma)
	clustered_ODM = gravity_ODM(df, d_max, level, gamma)
	epsilon = epsilon_matrix(combined_ODM, clustered_ODM)
	return epsilon