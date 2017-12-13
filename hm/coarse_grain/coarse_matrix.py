import numpy as np

def reorder_matrix(M, clusters):
	'''
	Takes an origin-destination matrix M and an array of cluster indices to which each location belongs and reorders M with clustered locations adjacent to eachother

	'''

	clusters = np.array(clusters)
	nOrder = np.argsort(clusters) 
	nM = M[:, nOrder][nOrder]

	return nM, nOrder

def reduce_matrix(M, clusters):
	'''
	Takes an ordered-by-cluster OD matrix M and returns the reduced cluster matrix with summed clusters of dimensions of len(clusters) x len(clusters)

	'''
	nDim = len(np.unique(clusters))
	clusterLengths = np.unique(clusters, return_counts=True)[1]
	cumClusterLengths = np.cumsum(clusterLengths)
	# hBlocks = np.hsplit(M, cumClusterLengths)[:-1]

	reducedM = []

	for i, vclusterL in enumerate(cumClusterLengths):
		for j, hclusterL in enumerate(cumClusterLengths):
			block = M[vclusterL -  clusterLengths[i]: vclusterL, hclusterL - clusterLengths[j]: hclusterL]
			reducedM.append(np.sum(block))

	reducedM = np.array(reducedM).reshape((nDim,nDim))

	return reducedM

def coarse_grain(M, clusters):
	'''
	Combines the action of reorder_matrix and reduce_matrix to coarse grain the system according to clusters

	'''
	return reduce_matrix(reorder_matrix(M, clusters)[0], clusters)


