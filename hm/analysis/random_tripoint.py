from hm.hm_models.gravity import gravity
from hm.pop_models.pop_random import random as pop_random
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from scipy import interpolate


N = 100
alpha, beta = 1, 1
gamma = 0.2
p = pop_random(N)
g = gravity(p, alpha, beta, gamma)

def neighbours(k):
	'''Returns a list of nearest neighobours pairs as a Numpy array.'''

	neighbours = []
	for i in range(p.size):
		distance = []
		for j in range(p.size):
			if i != j:
				distance.append(p.r(i, j))
		n = distance.index(min(distance))
		if i < j:
			neighbours.append([i, n+1])
		if i > j:
			neighbours.append([i, n])
	return np.array(neighbours[1:])

def neighbours_dist(k):
	'''
	Returns the (scaled) distance between two nearest neighbours as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	distance = []
	for i in neighbours(k):
		distance.append((p.r(i[0], i[1]))/np.sqrt(N))

	return np.array(distance)

def target_dist(i):
	'''
	Returns the (scaled) distance between the midpoint between two nearest neighbours and the
	target location i as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	r = []
	for n in neighbours(i):
		x = (p.locCoords[n[0]][0]+ p.locCoords[n[1]][0])*0.5
		y = (p.locCoords[n[0]][1]+ p.locCoords[n[1]][1])*0.5
		r.append((disp(p.locCoords[i], np.array([x, y])))/np.sqrt(N))
	return np.array(r)


def epsilon(i):
	'''Using abs!!!'''
	epsValues = []
	for n in neighbours(i):
		j, k = n[0], n[1]
		p2 = copy.deepcopy(p)
		if p2.locCoords[j][1] - p2.locCoords[k][1] > 0:
			p2.locCoords[j][1] = p2.locCoords[j][1] - 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
		else:
			p2.locCoords[j][1] = p2.locCoords[j][1] + 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
		if p2.locCoords[j][0] - p2.locCoords[k][0] > 0:
			p2.locCoords[j][0] = p2.locCoords[j][0] - 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])
		else:
			p2.locCoords[j][0] = p2.locCoords[j][0] + 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])
		p2.locCoords[k][1] = p2.locCoords[j][1]
		p2.locCoords[k][0] = p2.locCoords[j][0]
		g2 = gravity(p2, alpha, beta, gamma)
		eps = ((g2.flux(i, j) + g2.flux(i,k)) - (g.flux(i, j)+g.flux(i, k)))/(g2.flux(i, j) + g2.flux(i,k))
		epsValues.append(abs(eps))
	return np.array(epsValues)


def heatmap(i):
	x = target_dist(i)
	y = neighbours_dist(i)
	epsVals = epsilon(i)
	xy = np.column_stack((x, y))

	f = interpolate.griddata(xy, epsVals, (np.linspace(0,1, 0.1), np.linspace(0, 1, 0.1)), method = 'nearest')

	ax = sns.heatmap(epsVals)
	plt.show()
	return x, y

def neighbours_dist_plot():
	for i in range(p.size):
		plt.plot(neighbours_dist(i), epsilon(i), '.')
	plt.xlabel('$\~r_{jk}$')
	plt.ylabel('$\epsilon$')
	plt.show()