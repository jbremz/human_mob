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
		self.clusters = self.find_clusters()
		self.clusters_num = self.clusters_num(threshold)
		self.clustered_loc = self.get_clusters()
	
	def dist_matrix(self):
		pop = self.pop
		xy = pop.locCoords
		distance = pdist(xy)
		return distance
	
	def find_clusters(self):
		return hier.fcluster(linkage(self.dist_matrix), self.threshold, criterion = 'distance')
	
	def viz_clusters(self):
		pop = self.pop
		xy = pop.locCoords
		plt.scatter(xy[:,0], xy[:,1], c = self.clusters)  
		plt.show()
		
	def clusters_num(self, threshold):
		return len(np.unique(self.clusters))
	
	def get_clusters(self):
		locations = []
		groups = []
		for i in self.clusters:
			if i not in groups:
				locations.append([i,np.where(self.clusters == i)])
		locations = sorted(locations)
		for i in locations:
			groups.append(i[1])
		return groups
	
	def merge_population(self):
		summed_pop = []
		for i in self.clustered_loc:
			cluster_pop = []
			for loc in i:
				cluster_pop = self.pop.popDist[loc]
			summed_pop.append(sum(cluster_pop))
		return summed_pop