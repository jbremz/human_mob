from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.pop_models.pop_random import random as pop_random

N = 20
alpha, beta = 1, 1
gamma = 0.2
p = pop_random(N)
g = gravity(p, alpha, beta, gamma)

def neighbours():
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
			neighbours.append([i, n+1])
	return np.array(neighbours)


def eps():
	for i in range(p.size):
		for b in neighbours():
			j, k = b[0], b[1]
			if p.r(j, k) <= 0.5*np.sqrt(p.r(i, b[0]):
				print(i, j, k)
