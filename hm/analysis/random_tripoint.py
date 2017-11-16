from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
import numpy as np
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from scipy import stats


def neighb(p, i):
	'''Returns a list of the nearest neighobours of a given location i as a Numpy array.'''

	neighbours = []
	distance = []
	js = []
	for j in range(p.size):
		if j != i:
			distance.append(p.r(i, j))
			js.append(j)
	n = distance.index(min(distance))
	neighbours.append([i, js[n]])

	return np.array(neighbours)

def neighbours(p, k):
	'''
	Returns a list of all nearest neighbours pairs - excluding the target location k -
	as a Numpy array,
	'''

	neighbours = []
	for i in range(p.size):
		if i != k:
			if k not in neighb(p, i):
				neighbours.append(neighb(p, i)[0])

	return np.array(neighbours)

def r_jk(p, i):
	'''
	Returns the distance between all pairs of nearest neighbours - excluding the
	target location i - as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	distance = []
	for n in neighbours(p, i):
		distance.append((p.r(n[0], n[1])))

	return np.array(distance)

def r_ib(p, i):
	'''
	Returns the distance between the midpoint between two nearest neighbours and the
	target location i as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	r = []
	for n in neighbours(p, i):
		x = (p.locCoords[n[0]][0]+ p.locCoords[n[1]][0])*0.5
		y = (p.locCoords[n[0]][1]+ p.locCoords[n[1]][1])*0.5
		r.append((disp(p.locCoords[i], np.array([x, y]))))
	return np.array(r)

def epsilon(p, i, model, tilde = False):
	'''
	Returns epsilon for a given target location i.
	'''

	epsValues = []
	for n in neighbours(p, i):
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
			eps = (g2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(g2.flux(i, b))

		if isinstance(model, radiation):
			r2 = radiation(p2)
			eps = (r2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(r2.flux(i, b))

		if isinstance(model, opportunities):
			gamma = model.gamma
			o2 = opportunities(p2, gamma)
			eps = (o2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(o2.flux(i, b))

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
	return x, y

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
			for i in x:
				#only if m = 1 for all!
				gamma = model.gamma
				eps_values = 1 - (np.exp(-gamma*(np.sqrt(x**2 + (r_jk/2)**2)-x)))
				#eps_values = 1-(1-(np.arctan(r_jk/(2*x)))/(np.pi))*np.exp(model.gamma*(x-np.sqrt(x**2 + (r_jk/2)**2)))
	if isinstance(model, radiation):
		x = eps_vs_target(p, model, tilde)[0]
		for i in x:
			#only if m = 1 for all!
			eps_values = 1 - (3 + np.pi*p.size*x**2)/(2 + 3 + np.pi*p.size*(x**2 + (r_jk/2.)**2))
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
				#eps_values = 1-(1-(np.arctan(r_jk/(2*x)))/(np.pi))*np.exp(model.gamma*(x-np.sqrt(x**2 + (r_jk/2)**2)))
			return eps_values

def r_jk_plot(p, model, r_ib, tilde = False):
	'''
	Plots epsilon as a function of r_jk, given a costant value of r_ib.
	'''
	x = eps_vs_neighbours(p, model, tilde)[0]
	y = eps_vs_neighbours(p, model, tilde)[1]
	plt.plot(x*np.sqrt(p.size), y, '.', label = 'simulation')
	plt.plot(x*np.sqrt(p.size), eps_rjk(p, model, r_ib), '.', label = 'theory')
	linregr = stats.linregress(x, y)
	#plt.plot(x*np.sqrt(p.size), x*np.sqrt(p.size)*linregr[0] +linregr[1], '.', label = 'regression')
	plt.legend()
	plt.xlabel('$\~r_{jk}$')
	plt.ylabel('$\epsilon$')
	plt.show()

def r_ib_plot(p, model, r_jk, tilde = False):
	'''
	Plots epsilon as a function of r_ib, given a costant value of r_jk.
	'''

	x = eps_vs_target(p, model, tilde)[0, :]
	y = eps_vs_target(p, model, tilde)[1,:]
	step = int(p.size/2)
	for i in np.arange(0, len(x), step):
		mean_y = np.mean(y[i:i+step])
		plt.plot(x[i]*np.sqrt(p.size), mean_y, '.',)
	#plt.plot(x*np.sqrt(p.size), y, '.', label = 'simulation')
	plt.plot(x*np.sqrt(p.size), eps_rib(p, model, r_jk), '.', label = 'theory')
	plt.xlabel('$\~r_{ib}$')
	plt.ylabel('$\epsilon$')
	plt.legend()
	plt.show()

def test_ratio(p, model):
	#delta_theta = 2*np.arctan(r_jk/(2*r_ib))
	#integral = ((2*np.pi- delta_theta)/model.gamma**2) * (model.gamma*np.sqrt(1./p.size))
	r_ij = np.arange(0., 1., 0.01)
	r_jk = 0.1
	r_ib = np.sqrt(r_ij**2 - (r_jk/2.)**2)
	ratio = (np.exp(-model.gamma*r_ij))/(np.exp(-model.gamma*r_ib))
	plt.plot(r_ij, ratio)
	plt.show()
