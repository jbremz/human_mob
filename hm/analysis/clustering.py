import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy as hier
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

class Clusters:
	def __init__(self, pop, threshold):
		self.pop = pop
		self.threshold = threshold
		self.dist_matrix = self.dist_matrix()
		self.clusters = self.find_clusters() # Hello Jim: USE THIS!
		self.clusters_num = self.clusters_num(threshold)
		self.clustered_loc = self.get_clusters()
		self.clustered_pop = self.merge_population()
	
	def dist_matrix(self):
		'''Returns the condensed distance matrix as a 1d numpy array.''' 
		pop = self.pop
		xy = pop.locCoords
		distance = pdist(xy)
		return distance
	
	def find_clusters(self):
		'''Returns flat clusters from the hierarchical clustering.'''
		return hier.fcluster(linkage(self.dist_matrix,method = 'centroid'), self.threshold, criterion = 'distance')
	
	def viz_clusters(self):
		'''Plots locations with colors distinguishing clusters.''' 
		pop = self.pop
		xy = pop.locCoords
		plt.scatter(xy[:,0], xy[:,1], c = self.clusters)  
		plt.show()
		
	def clusters_num(self, threshold):
		'''Returns the number of clusters formed.'''
		return len(np.unique(self.clusters))
	
	def get_clusters(self):
		'''
		Returns numpy array containing lists of locations within each cluster.
		
		self.get_cluster[n] returns a list of locations in the nth cluster.
		'''
		locations = []
		groups = []
		for i in self.clusters:
				locations.append([i, np.where(self.clusters == i)[0]])
		locations = sorted(locations, key=lambda x: x[0])
		for i in locations:
			if list(i[1]) not in groups:
				groups.append(list(i[1]))
		return np.array(groups)
	
	def which_cluster(self, loc):
		'''Return index of cluster which loc belongs to.'''
		return self.clusters[loc]
	
	def merge_population(self):
		'''Returns numpy array with the total population in each cluster.'''
		summed_pop = []
		for i in self.clustered_loc:
			cluster_pop = []
			for loc in i:
				cluster_pop.append(self.pop.popDist[loc])
			for populations in cluster_pop:
				if sum(cluster_pop) not in summed_pop:
					summed_pop.append(sum(cluster_pop))
		return np.array(summed_pop)
	
	#def clusters_loc(self):
		