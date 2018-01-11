import numpy as np 

def rms_eps(eps_matrix):
	'''
	Returns mean of epsilon matrix (ignoring diagonal)

	'''
	mask = np.ones(eps_matrix.shape, dtype=bool)
	np.fill_diagonal(mask, 0)
	return np.sqrt(np.mean(eps_matrix[mask]**2))

def sigma_eps(eps_matrix):
	'''
	Returns standard deviation of epsilon matrix (ignoring diagonal)

	'''
	mask = np.ones(eps_matrix.shape, dtype=bool)
	np.fill_diagonal(mask, 0)
	return np.std(eps_matrix[mask])