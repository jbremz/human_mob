import numpy as np 
from hm.coarse_grain.clustering import Clusters
from hm.pop_models.pop_explicit import explicit as pop_explicit
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.coarse_grain.coarse_matrix import epsilon_matrix
from hm.coarse_grain.coarse_matrix import coarse_grain as coarse_grain_matrix
from hm.analysis.scaling.pop_tools import make_pop
from hm.utils.utils import gamma_est
from hm.pop_models import pop_random
import pandas


class pop_hier:
	'''
	Takes a dataframe (of the CDRC dataset form) and d_max: a list of the maximum distance between clusters at each desired level of clustering

	Contains methods to produce epsilon matrices at different levels of clustering.
	
	'''

	def __init__(self, df, d_maxs):
		self.df = df
		if isinstance(df, pandas.DataFrame):
			self.pop = make_pop(df)
		if isinstance(df, pop_random.random):
			self.pop = df
		self.d_maxs = d_maxs
		self.levels = self.iterate(self.d_maxs) # the list of cluster objects at each level defined in d_maxs
		self.original_ODM = False # the ODM of Level 0 of the hierarchy (defined once reduced_ODM is called for the first time)

	def iterate(self, d_maxs):
		"""Returns a list of Clusters objects with all the levels up to specified level."""

		# Clustering starts at level 1 (level 0 is unclustered)
		if len(d_maxs) < 1:
			print("iterate() only accepts clustering levels > 0")
			return

		levels = []

		for d_max in d_maxs:
			clusters = Clusters(self.pop, d_max)
			levels.append(clusters)
			
		return levels	

	def cluster_population(self, cluster, pw = True):
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

	def gravity_ODM(self, level, gamma=False, exp=False):
		"""
		Returns the ODM for the gravity model at a specific level of clustering.
		
		If level == 0 (no clustering), a dataframe needs to be specified so that 
		a population object (the original) can be created from it.
		"""
		if level == 0:
			pop = self.levels[0].pop
			if gamma is bool:
				S = np.mean(self.df['Area']) # mean population unit area
		
		else:
			clustering = self.levels[level-1]
			pop = self.cluster_population(clustering)
			if gamma is bool:
				S = np.mean(self.levels[level-1].clustered_area) # mean population unit area
		
		if gamma is bool:
			gamma = gamma_est(S, exp=exp) # calculate the gamma exponent with the average population unit area
		g = gravity(pop, 1, 1, gamma, exp=exp)
		return g.ODM()

	def reduced_ODM(self, level, exp=False):
		"""Returns ODM for the combined flow between locations."""

		if type(self.original_ODM) == bool:
			self.original_ODM = self.gravity_ODM(level=0, exp=exp)

		if level != 0:
			self.original_ODM
			clust = self.levels[level-1].clusters
			reduced_ODM = coarse_grain_matrix(self.original_ODM, clust)
		else: # not reduced
			reduced_ODM = self.original_ODM

		return reduced_ODM

	def DM_level(self, level):
		'''
		Returns the distance matrix at a given level

		'''
		if level == 0:
			return self.pop.DM

		if level > 0:
			return self.cluster_population(self.levels[level-1]).DM

	def epsilon(self, level, model=False, exp=False):
		"""Returns the epsilon matrix (defined compared to the ODM at the original location resolution) at a specific level of clustering."""
		
		if type(model) is not bool:
			print("We've only included gravity model so far, sorry x")

		if level > len(self.d_maxs):
			print("Object has only been initialised with " + str(len(self.d_maxs)) + " levels")
			return

		clustered_ODM = self.gravity_ODM(level, exp=exp)
		combined_ODM = self.reduced_ODM(level, exp=exp)
		epsilon = epsilon_matrix(combined_ODM, clustered_ODM)

		return epsilon



