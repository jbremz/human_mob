import numpy as np 
from hm.analysis.scaling.pop_hierarchy import pop_hier
from scipy.optimize import minimize_scalar
from scipy.optimize import leastsq
from scipy.optimize import minimize


def optimise_eps(hier, level, gamma_0, exp = False):
	# if not isinstance(hier, pop_hier):
	# 	raise NameError("hier must be an object of the pop_hier class!")
	
	def mean(gamma):
		return mean_eps(hier, level, gamma_0, gamma, exp=exp)
	
	#array = np.array([mean(gamma_0), mean(gamma_0+1)])
	#opt = leastsq(mean, gamma_0)
	opt = minimize(mean, gamma_0, tol = 0.001)
	#opt = minimize_scalar(mean, bounds=(gamma_0-3, gamma_0+3), tol = 0.001)
	
	return opt
		
def mean_eps(hier, level, gamma_0, gamma, exp=False):
	eps = hier.epsilon_to_opt(level, gamma_0, gamma, exp=exp)
	N = len(eps)
	grid = np.indices((N, N))
	diag = grid[0] == grid[1]
	eps[np.where(diag==False)] #take diagonal out and flatten array
	
	mean = np.sqrt(np.mean(np.square(eps)))
	
	return mean