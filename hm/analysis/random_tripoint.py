from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
import numpy as np
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error as mse


def neighbours(p):
	'''
	Returns array with distances and indices of all pairs of nearest neighours.
	'''
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(p.locCoords)
	distances, indices = nbrs.kneighbors(p.locCoords)
	return distances, indices

def r_jk(p, i):
	'''
	Returns the distance between all pairs of nearest neighbours as a Numpy array
	- excludint target.
	'''
	distance = []
	index = np.where(neighbours(p)[1][:,0] != i)
	index_2 = np.where(neighbours(p)[1][:,1] != i)
	indices = np.intersect1d(index, index_2)
	for item in indices:
		distance.append(neighbours(p)[0][item][1])
	return np.array(distance)

def r_ib(p, i):
	'''
	Returns the distance between the midpoint between two nearest neighbours and the
	target location i as a Numpy array.
	'''

	distance = []
	for n in neighbours(p)[1]:
		if n[0] != i and n[1] != i:
			x = (p.locCoords[n[0]][0]+ p.locCoords[n[1]][0])*0.5
			y = (p.locCoords[n[0]][1]+ p.locCoords[n[1]][1])*0.5
			distance.append((disp(p.locCoords[i], np.array([x, y]))))
	return np.array(distance)

def epsilon(p, i, model, exp=True, tilde = False):
	'''
	Returns epsilon for a given target location i.
	'''
	epsValues = []
	for n in neighbours(p)[1]:
		if n[0] != i and n[1] != i:
			j, k = n[0], n[1]
			p2 = copy.deepcopy(p)

			if tilde == True:
				# use m tilde correction
				p2.popDist[j] = p2.popDist[k] + p2.popDist[j] - model.flux(j, k) - model.flux(k, j)
			if tilde == False:
				# merge two populations
				p2.popDist[j] = p2.popDist[k] + p2.popDist[j]

			# move j to midpoint
			p2.locCoords[j][0] = 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])
			p2.locCoords[j][1] = 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
			p2.DM = p2.distance_matrix()

			p2.popDist[k] = 0. #remove k
			b = j #rename j

			if isinstance(model, gravity):
				alpha = model.alpha
				beta = model.beta
				gamma = model.gamma

				g2 = gravity(p2, alpha, beta, gamma, exp=exp)
				flow_ib = g2.flux(i, b)
				eps = (flow_ib - (model.flux(i, j)+model.flux(i, k)))/(flow_ib)

			if isinstance(model, radiation):
				r2 = radiation(p2)
				flow_ib = r2.flux(i, b)
				eps = (flow_ib - (model.flux(i, j)+model.flux(i, k)))/(flow_ib)

			epsValues.append(abs(eps))

	return np.array(epsValues)

def eps_vs_neighbours(p, model, exp = True, tilde = False):
	'''
	Returns two arrays with r_jk between all possible target-pair pairs (sorted
	in ascending order) and the corresponding epsilon.
	'''

	x = []
	y = []
	for i in range(p.size):
		x.append(r_jk(p, i))
		y.append(epsilon(p, i, model, exp=exp, tilde=tilde))
	x = np.concatenate(x)
	y = np.concatenate(y)
	xy = np.array([x, y])
	xy = xy[:,np.argsort(xy[0])]
	return xy

def eps_vs_target(p, model, exp=True, tilde = False):
	'''
	Returns two arrays with all possible r_ib (sorted in ascending order) and
	the corresponding epsilon.
	'''

	y = []
	x = []
	for i in range(p.size):
		x.append(r_ib(p, i))
		y.append(epsilon(p, i, model, exp = exp, tilde=tilde))
	x = np.concatenate(x)
	y = np.concatenate(y)
	xy = np.array([x, y])
	xy = xy[:,np.argsort(xy[0])]
	return xy

def eps_rib(p, model,r_jk, exp=True, tilde = False):
	'''
	Returns the analytical form of epsilon as a function of r_ib for the
	exponential gravity model.
	'''

	if isinstance(model, gravity):
		x = eps_vs_target(p, model,exp=exp, tilde=tilde)[0]
		r_ij = np.sqrt(x**2 + (r_jk/2)**2)
		gamma = model.gamma
		
		if model.exp:
			for i in x:
				#only if m = 1 for all!
				eps_values = 1 - (np.exp(-gamma*(r_ij-x)))
				
		else:
			#only if m = 1 for all!
			eps_values = 1 - (r_ij/x)**(-gamma)
			
	if isinstance(model, radiation):
		x = eps_vs_target(p, model, exp=exp, tilde=tilde)[0]
		for i in x:
			#only if m = 1 for all!
			eps_values = 1 - ((x**2)*(np.pi*p.size*x**2 + 2))/((x**2 + (r_jk/2.)**2)*(np.pi*p.size*(x**2 + (r_jk/2.)**2 )+1))
	return eps_values

def eps_rjk(p, model,r_ib, exp=True, tilde = False):
	'''
	Returns the analytical form of epsilon as a function of r_jk for the
	exponential gravity model.
	'''

	if isinstance(model, gravity):
		x = eps_vs_neighbours(p, model,exp=exp, tilde=tilde)[0]
		r_ij = np.sqrt(r_ib**2 + (x/2)**2)
		
		for i in x:
			#only if m = 1 for all!
			gamma = model.gamma
			if model.exp:
				eps_values = 1 - (np.exp(-gamma*(r_ij - r_ib)))
			else:
				eps_values = 1 - (r_ij/r_ib)**(-gamma)
			

	if isinstance(model, radiation):
		x = eps_vs_neighbours(p, model,exp=exp, tilde=tilde)[0]
		r_ij = np.sqrt(r_ib**2 + (x/2)**2)
		rho = p.size
		for i in x:
			#only if m = 1 for all!
			eps_values = 1 - (r_ib**2/r_ij**2)*(np.pi * rho * r_ib**2 + 2)/(np.pi * rho * r_ij**2 + 1)

	return eps_values

def rjk_binning(p, model, exp):
	x,y = eps_vs_neighbours(p, model, exp=exp)
	mean_y = []
	std_y = []
	mean_x, indices = np.unique(eps_vs_neighbours(p, model,exp=exp)[0], return_index=True)
	for i in range(1, len(indices)+1):
		if i == len(indices):
			ran = list(range(indices[-1], len(x)))
		else:
			ran = list(range(indices[i-1], indices[i]+1))
		mean_y.append(np.mean(y[ran[0]:ran[-1]]))
		std_y.append(np.std(y[ran[0]:ran[-1]])/np.sqrt(len(y[ran[0]:ran[-1]])))

	return x, mean_x, mean_y, np.array(std_y)
	
def mean_r(p, model, x_value, exp=True, tilde = False):
	if x_value == "r_ib":
		eps_target = eps_vs_target(p, model, exp=exp, tilde=tilde)
		x = eps_target[0, :]
		y = eps_target[1,:]
		
		step = int(len(y)/(p.size*2))
		mean_y = []
		mean_x = []
		std_y = []
		for i in np.arange(0, int(len(x)/6), 40):
			if i >= step:
				mean_y.append(np.mean(y[i-step:i+step]))
				std = np.std(y[i:i+int(step/2)])
				std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
				mean_x.append(np.mean(x[i:i+int(step/2)]))
			if i < step:
				mean_y.append(np.mean(y[i:i+int(step/2)]))
				std = np.std(y[i:i+int(step/2)])
				std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
				mean_x.append(np.mean(x[i:i+int(step/2)]))
				
		for i in np.arange(int(len(x)/6), len(x), 50):
			#used 50 for exp
			if i >= step:
				mean_y.append(np.mean(y[i-step:i+step]))
				std = np.std(y[i:i+int(step/2)])
				std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
			if i < step:
				mean_y.append(np.mean(y[i:i+int(step/2)]))
				std = np.std(y[i:i+int(step/2)])
				std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
				mean_x.append(np.mean(x[i:i+int(step/2)]))
		
	if x_value == "r_jk":
		if isinstance(model, gravity):
			x, mean_x, mean_y, std_y = rjk_binning(p, model, exp=exp)
			
		if isinstance(model, radiation):
			x, x_new, y, std_old = rjk_binning(p, model, exp=exp)
			
			step = int(len(y)/5)
			mean_y = []
			mean_x = []
			std_y = []
			
			for i in np.arange(0, int(len(x)), 1):
				if i >= step:
					mean_y.append(np.mean(y[i-step:i+step]))
					std = np.std(y[i:i+int(step/2)])
					std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
					mean_x.append(np.mean(x_new[i:i+int(step/2)]))
				#std_y = std_old + np.array(std_y)
					
				if i < step:
					mean_y.append(np.mean(y[i:i+int(step/2)]))
					std = np.std(y[i:i+int(step/2)])
					std_y.append(std/np.sqrt(len(y[i:i+int(step/2)])))
					mean_x.append(np.mean(x_new[i:i+int(step/2)]))

	return x, np.array(mean_x), mean_y, std_y

def plot_ratio(p, model, r_jk, collapse = False, tilde = False):
	theory = eps_rib(p, model, r_jk)
	mean = mean_r(p, model,"r_ib")
	x = mean[0]
	mean_y = mean[1]
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	# Resolution
	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi = 500)
	
	if collapse == True:
		plt.plot(x, mean_y/theory, '.', label = 'N = '+str(p.size))
		plt.xlabel('$\~r_{ib}} / sqrt(N)} $')
	else:
		plt.plot(x*np.sqrt(p.size), mean_y/theory, '.', label = 'N = '+str(p.size))
		plt.xlabel('$\~r_{ib}$')
		
	plt.ylabel('$\epsilon_{sim}$/$\epsilon_{ana}$', fontsize = 20)
	# Legend
	plt.legend(frameon=False, fontsize=20)
	
	# Axes/tick labels
	plt.tick_params(axis='both', labelsize=15)
	plt.ticklabel_format(style='sci')
	
	plt.tight_layout()
	
	plt.savefig('ratio')


def plot_ratio_rjk(p, model, r_ib, tilde = False, collapse = False):
	theory = eps_rjk(p, model, r_ib)
	mean = mean_r_jk(p, model, r_ib)
	x =  mean[0]
	mean_y = mean[1]
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	if collapse == True:
		plt.plot(x, mean_y/theory, '.', label = 'N = '+str(p.size))
		plt.xlabel('$\~r_{jk}} / sqrt(N)} $')
	else:
		plt.plot(x*np.sqrt(p.size), mean_y/theory, '.', label = 'N = '+str(p.size))
		plt.xlabel('$\~r_{jk}$')
	plt.ylabel('ratio')
	plt.legend()
