from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
import numpy as np
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chisquare

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

def epsilon(p, i, model, tilde = False):
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

			p2.popDist[k] = 0. #remove k
			b = j #rename j

			if isinstance(model, gravity):
				alpha = model.alpha
				beta = model.beta
				gamma = model.gamma

				g2 = gravity(p2, alpha, beta, gamma, exp=True)
				flow_ib = g2.flux(i, b)
				eps = (flow_ib - (model.flux(i, j)+model.flux(i, k)))/(flow_ib)

			if isinstance(model, radiation):
				r2 = radiation(p2)
				flow_ib = r2.flux(i, b)
				eps = (flow_ib - (model.flux(i, j)+model.flux(i, k)))/(flow_ib)

			epsValues.append(abs(eps))

	return np.array(epsValues)

def eps_vs_neighbours(p, model, tilde = False):
	'''
	Returns two arrays with r_jk between all possible target-pair pairs (sorted
	in ascending order) and the corresponding epsilon.
	'''

	x = []
	y = []
	for i in range(p.size):
		x.append(r_jk(p, i))
		y.append(epsilon(p, i, model, tilde))
	x = np.concatenate(x)
	y = np.concatenate(y)
	xy = np.array([x, y])
	xy = xy[:,np.argsort(xy[0])]
	return xy

def eps_vs_target(p, model, tilde = False):
	'''
	Returns two arrays with all possible r_ib (sorted in ascending order) and
	the corresponding epsilon.
	'''

	y = []
	x = []
	for i in range(p.size):
		x.append(r_ib(p, i))
		y.append(epsilon(p, i, model, tilde))
	x = np.concatenate(x)
	y = np.concatenate(y)
	xy = np.array([x, y])
	xy = xy[:,np.argsort(xy[0])]
	return xy

def eps_rib(p, model,r_jk, tilde = False):
	'''
	Returns the analytical form of epsilon as a function of r_ib for the
	exponential gravity model.
	'''

	if isinstance(model, gravity):
		if model.exp:
			x = eps_vs_target(p, model, tilde)[0]
			#unnecessary loop below!!!!!!!! CHANGE
			for i in x:
				#only if m = 1 for all!
				gamma = model.gamma
				eps_values = 1 - (np.exp(-gamma*(np.sqrt(x**2 + (r_jk/2)**2)-x)))
	if isinstance(model, radiation):
		x = eps_vs_target(p, model, tilde)[0]
		for i in x:
			#only if m = 1 for all!
			#changed today
			eps_values = 1 - ((x**2)*(np.pi*p.size*x**2 + 2))/((x**2 + (r_jk/2.)**2)*(np.pi*p.size*(x**2 + (r_jk/2.)**2 )+1))
	return eps_values

def eps_rjk(p, model,r_ib, tilde = False):
	'''
	Returns the analytical form of epsilon as a function of r_jk for the
	exponential gravity model.
	'''

	if isinstance(model, gravity):
		if model.exp:
			x = eps_vs_neighbours(p, model, tilde)[0]
			for i in x:
				#only if m = 1 for all!
				gamma = model.gamma
				eps_values = 1 - (np.exp(-gamma*(np.sqrt(r_ib**2 + (x/2)**2)-r_ib)))

	if isinstance(model, radiation):
		x = eps_vs_neighbours(p, model, tilde)[0]
		for i in x:
			#only if m = 1 for all!
			eps_values = 1 - ((r_ib**2)*(np.pi*p.size*r_ib**2 + 2))/((r_ib**2 + (x/2.)**2)*(np.pi*p.size*(r_ib**2 + (x/2.)**2 )+1))

	return eps_values

def r_jk_plot(p, model, r_ib, tilde = False):
	'''
	Plots epsilon as a function of r_jk, given a costant value of r_ib.
	'''

	x = mean_r_jk(p, model, r_jk, tilde = False)[0]
	mean_y = mean_r_jk(p, model, r_ib, tilde = False)[1]
	plt.plot(x*np.sqrt(p.size), mean_y, '.', label = 'simulation')
	plt.plot(x*np.sqrt(p.size), eps_rjk(p, model, r_ib), '.', label = 'theory')
	plt.legend()
	plt.xlabel('$\~r_{jk}$')
	plt.ylabel('$\epsilon$')
	plt.show()

def mean_r_ib(p, model, r_jk, tilde = False):
	eps_target = eps_vs_target(p, model, tilde)
	x = eps_target[0, :]
	y = eps_target[1,:]
	step = int(len(y)/60)
	mean_y = []
	for i in np.arange(0, len(x), 1):
		if i >= step:
			mean_y.append(np.mean(y[i-step:i+step]))
		if i < step:
			mean_y.append(np.mean(y[i:i+int(step/2)]))
	return x, mean_y

def mean_r_jk(p, model, r_jk, tilde = False):
	eps_neighbours = eps_vs_neighbours(p, model, tilde)
	x = eps_neighbours[0, :]
	y = eps_neighbours[1,:]
	step = int(len(y)/60)
	mean_y = []
	for i in np.arange(0, len(x), 1):
		if i >= step:
			mean_y.append(np.mean(y[i-step:i+step]))
		if i < step:
			mean_y.append(np.mean(y[i:i+int(step/2)]))
	return x, mean_y

def r_ib_plot(p, model, r_jk, tilde = False):
	'''
	Plots epsilon as a function of r_ib, given a costant value of r_jk.
	'''

	#plt.plot(x*np.sqrt(p.size), y, '.', label = 'simulation')
	x = mean_r_ib(p, model, r_jk, tilde = False)[0]
	mean_y = mean_r_ib(p, model, r_jk, tilde = False)[1]
	plt.plot(x*np.sqrt(p.size), eps_rib(p, model, r_jk), '.', label = 'theory')
	plt.plot(x*np.sqrt(p.size), mean_y, '.')
	plt.xlabel('$\~r_{ib}$')
	plt.ylabel('$\epsilon$')
	plt.legend()
	plt.show()

def plot_ratio(p, model, r_jk, tilde = False):
	eps_target = eps_vs_target(p, model, tilde)
	x = eps_target[0, :]
	y = eps_target[1,:]
	step = int(len(y)/20)
	theory = eps_rib(p, model, r_jk)
	mean_y = []
	for i in np.arange(0, len(x), 1):
		if i < step:
			mean_y.append(p.size*np.mean(y[i:i+int(step/2)]))
		if i >= step:
			mean_y.append(p.size*np.mean(y[i-step:i+step]))
	#plt.plot(x*np.sqrt(p.size), mean_y/theory, '.')
	plt.plot(x, mean_y/theory, '.')
	plt.ylabel('ratio')
	plt.xlabel('$\~r_{ib}$')