from hm.hm_models.gravity import gravity
from hm.pop_models.pop_random import random as pop_random
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
from hm.utils.utils import disp
from scipy import interpolate
from itertools import combinations


N = 30
alpha, beta = 1, 1
gamma = 2.
p = pop_random(N)
g = gravity(p, alpha, beta, gamma)

def neighb(i):
	'''Returns a list of nearest neighobours pairs as a Numpy array.'''

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

def neighbours(k):
	neighbours = []
	for i in range(p.size):
		if i != k:
			neighbours.append(neighb(i)[0])

	return np.array(neighbours)

def neighbours_dist(k):
	'''
	Returns the (scaled) distance between two nearest neighbours as a Numpy array.
	The pair's index matches that of neighbours().
	'''

	distance = []
	for i in neighbours(k):
		distance.append((p.r(i[0], i[1])))

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
		r.append((disp(p.locCoords[i], np.array([x, y]))))
	return np.array(r)

def epsilon(i):
	'''Using abs!!!'''
	epsValues = []
	for n in neighbours(i):
		j, k = n[0], n[1]
		p2 = copy.deepcopy(p)

		#move j to midpoint

		p2.locCoords[j][1] = 0.5*(p2.locCoords[j][1]+ p2.locCoords[k][1])
		p2.locCoords[j][0] = 0.5*(p2.locCoords[j][0]+ p2.locCoords[k][0])

		p2.popDist[j] = p2.popDist[k] + p2.popDist[j] #merge two populations
		p2.popDist[k] = 0. #remove k
		b = j #rename j

		g2 = gravity(p2, alpha, beta, gamma)
		eps = (g2.flux(i, b) - (g.flux(i, j)+g.flux(i, k)))/(g2.flux(i, b))
		epsValues.append(abs(eps))
	return np.array(epsValues)

def rev_epsilon(i):
	'''Using abs!!!'''
	epsValues = []
	for n in neighbours(i):
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
		epsValues.append(abs(eps))
	return np.array(epsValues)

def neighbours_dist_plot():
	for i in range(p.size):
		plt.plot(neighbours_dist(i)*np.sqrt(N), epsilon(i), '.')
	plt.xlabel('$\~r_{jk}$')
	plt.ylabel('$\epsilon$')
	plt.show()

def target_dist_plot():
	for i in range(p.size):
		plt.plot(target_dist(i)*np.sqrt(N), epsilon(i), '.')
	plt.xlabel('$\~r_{ib}$')
	plt.ylabel('$\epsilon$')
	plt.show()

'''def heatmap(i):
	x = target_dist(i)
	y = neighbours_dist(i)
	epsVals = epsilon(i)
	xy = np.column_stack((x, y))

	f = interpolate.griddata(xy, epsVals, (np.linspace(0,1, 0.1), np.linspace(0, 1, 0.1)), method = 'nearest')

	ax = sns.heatmap(epsVals)
	plt.show()
	return x, y'''
