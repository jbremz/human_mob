import numpy as np 
from hm.analysis.scaling.pop_hierarchy import pop_hier
from scipy.optimize import minimize_scalar

def optimise_eps(hier, level, gamma_0):
	if not isinstance(hier, pop_hier):
		raise NameError("hier must be an object of the pop_hier class!")
	
	def mean(gamma):
		return mean_eps(hier, level, gamma_0, gamma)
	
	opt = minimize_scalar(mean, bounds=(gamma_0-3, gamma_0+3), tol = 0.001)
		
	return opt
		
def mean_eps(hier, level, gamma_0, gamma):
	eps = hier.epsilon_to_opt(level, gamma_0, gamma)
	N = len(eps)
	grid = np.indices((N, N))
	diag = grid[0] == grid[1]
	eps[np.where(diag==False)] #take diagonal out and flatten array
	
	mean = np.sqrt(np.mean(np.square(eps)))
	
	return mean