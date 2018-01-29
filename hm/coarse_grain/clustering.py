import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy as hier
from scipy.cluster.hierarchy import linkage
from hm.pop_models.pop_explicit import explicit as pop_explicit

class Clusters:
	def __init__(self, pop, threshold):
		self.pop = pop
		self.threshold = threshold
		self.clusters = self.find_clusters() # Hello Jim: USE THIS!
		self.clusters_num = self.clusters_num(threshold)
		self.clustered_loc = self.get_clusters()
		self.clustered_pop = self.merge_population()
		if isinstance(self.pop, pop_explicit):
			self.clustered_area = self.merge_areas()
	
	def find_clusters(self):
		'''Returns flat clusters from the hierarchical clustering.'''
		flat_DM = self.pop.flat_DM
		return hier.fcluster(linkage(flat_DM, method = 'centroid'), self.threshold, criterion = 'distance')
	
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
			cluster_pop = 0
			for loc in i:
				cluster_pop += self.pop.popDist[loc]
			summed_pop.append(cluster_pop)
		return np.array(summed_pop)
	
	def merge_areas(self):
		'''Returns numpy array with the total area covered by each cluster.'''
		summed_area = []
		for i in self.clustered_loc:
			cluster_area = 0
			for loc in i:
				cluster_area += self.pop.locArea[loc]
			summed_area.append(cluster_area)
		return np.array(summed_area)
	
	def average_area(self):
		'''Return the average surface area.'''
		avg = sum(self.clustered_area)/self.clusters_num
		
		return avg
			
		
	def centroids(self):
		x_c = []
		y_c = []
		for i in self.clustered_loc:
			x = []
			y = []
			for loc in i:
				x.append(self.pop.locCoords[loc][0])
				y.append(self.pop.locCoords[loc][1])
			x_c.append(sum(x)/(len(x)))
			y_c.append(sum(y)/len(y))
		xy = np.array([x_c, y_c])
		return xy		
	
	def pw_centroids(self):
		x_c = []
		y_c = []
		index = 0
		for i in self.clustered_loc:
			x = []
			y = []
			for loc in i:
				x.append(self.pop.popDist[loc]*self.pop.locCoords[loc][0])
				y.append(self.pop.popDist[loc]*self.pop.locCoords[loc][1])
			M = self.clustered_pop[index]
			x_c.append(sum(x)/M)
			y_c.append(sum(y)/M)
			index += 1
		xy = np.array([x_c, y_c])
		return xy				