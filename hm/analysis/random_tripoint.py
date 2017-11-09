from hm.hm_models.gravity import gravity
from hm.pop_models.pop_random import random as pop_random
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from scipy.interpolate import griddata
from itertools import combinations
from scipy import stats


def neighb(p, i):
	'''Returns a list of the nearest neighobours of a given location as a Numpy array.'''

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
	Returns a list of all nearest neighbours pairs - excluding the target location -
	as a Numpy array,
	'''

	neighbours = []
	for i in range(p.size):
		if i != k:
			if k not in neighb(p, i):
				neighbours.append(neighb(p, i)[0])

	return np.array(neighbours)

def neighbours_dist(p, k):
	'''
	Returns the (scaled) distance between two nearest neighbours as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	distance = []
	for i in neighbours(p, k):
		distance.append((p.r(i[0], i[1])))

	return np.array(distance)

def target_dist(p, i):
	'''
	Returns the (scaled) distance between the midpoint between two nearest neighbours and the
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
	'''Using abs!!!'''
	epsValues = []
	for n in neighbours(p, i):
		j, k = n[0], n[1]
		p2 = copy.deepcopy(p)

		#move j to midpoint

		p2.locCoords[j][1] = 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
		p2.locCoords[j][0] = 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])

		#if tilde == True:
		#	p2.popDist[j] = p2.popDist[k] + p2.popDist[j] - p2.popDist[j]*g.flux(j, k) - p2.popDist[k]*g.flux(j, k)
		p2.popDist[j] = p2.popDist[k] + p2.popDist[j] #merge two populations

		p2.popDist[k] = 0. #remove k
		b = j #rename j

		if isinstance(model, gravity):
			alpha = model.alpha
			beta = model.beta
			gamma = model.gamma

			g2 = gravity(p2, alpha, beta, gamma)
			eps = (g2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(g2.flux(i, b))

		if isinstance(model, radiation):
			r2 = radiation(p2)
			eps = (r2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(r2.flux(i, b))

		if isinstance(model, opportunities):
			gamma = model.gamma
			o2 = opportunities(p2, gamma)
			eps = (o2.flux(i, b) - (model.flux(i, j)+model.flux(i, k)))/(o2.flux(i, b))
		epsValues.append(eps)

	return np.array(epsValues)

def rev_epsilon(p, g, i, tilde = False):
	'''Using abs!!!'''
	epsValues = []
	for n in neighbours(p, i):
		j, k = n[0], n[1]
		p2 = copy.deepcopy(p)

		#move j to midpoint

		p2.locCoords[j][1] = 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
		p2.locCoords[j][0] = 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])

		p2.popDist[j] = p2.popDist[k] + p2.popDist[j] #merge two populations
		p2.popDist[k] = 0. #remove k
		b = j #rename j

		g2 = gravity(p2, alpha, beta, gamma)
		eps = (g2.flux(b, i) - (g.flux(j, i)+g.flux(k, i)))/(g2.flux(b, i))
		epsValues.append(eps)
	return np.array(epsValues)

def neighbours_dist_plot(p, model, tilde = False):
	y = []
	x = []
	for i in range(p.size):
		y.append(epsilon(p, i, model, tilde))
		x.append(neighbours_dist(p, i))
	x = np.concatenate(x)
	y = np.concatenate(y)
	plt.plot(x*np.sqrt(p.size), y, '.')
	if isinstance(model, gravity):
		plt.plot(x*np.sqrt(p.size), np.arctan(x)/(np.pi), label = 'simulation')
		regress = stats.linregress(x*np.sqrt(p.size), y)
		plt.plot(x*np.sqrt(p.size), x*np.sqrt(p.size)*regress[0] + regress[1], label = 'theory')
		plt.legend()
	plt.xlabel('$\~r_{jk}$')
	plt.ylabel('$\epsilon$')
	plt.show()

def target_dist_plot(p, model, tilde = False):
	y = []
	x = []
	for i in range(p.size):
		y.append(epsilon(p, i, model, tilde))
		x.append(target_dist(p, i))
	x = np.concatenate(x)
	y = np.concatenate(y)
	plt.plot(x*np.sqrt(p.size), y, '.')
	plt.xlabel('$\~r_{ib}$')
	plt.ylabel('$\epsilon$')
	plt.show()

def contour():
	'''x = []
	y = []
	z = []
	for i in range(p.size):
		x.append(target_dist(i))
		y.append(neighbours_dist(i))
		z.append(epsilon(i))
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)'''
	xi = np.linspace(0,1., 30)
	yi = np.linspace(0,1., 30)
	# grid the data.
	#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
	X, Y = np.meshgrid(xi, yi)

	zi = 1 - (np.arctan(Y/X)/(2*np.pi))

	CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
	CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
	plt.colorbar() # draw colorbar
	# plot data points.
	#plt.scatter(x,y,marker='o',c='b',s=5)
