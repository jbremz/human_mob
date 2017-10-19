import importlib
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.pop_models.pop_random import random as pop_random
from hm.pop_models.pop_explicit import explicit as pop_explicit
from matplotlib import pyplot as plt
import numpy as np

popDist = [3,4,7,5,6]
locCoords = [[2,3],[3,2],[-5,9],[0,1],[1,-8]]
alpha, beta, K = 1, 1, 1
gamma = 0.2
N = 30

p = pop_random(N)
g = gravity(p, alpha, beta, gamma)
r = radiation(p)
o = opportunities(p, gamma)

distance = []
flux_gravity = []
flux_radiation = []
flux_opportunities = []
for i in range(p.size):
	for j in range(p.size):
		if i != j:
			distance.append(p.r(i, j))
			flux_gravity.append(g.flux(i, j))
			flux_radiation.append(r.flux(i, j))
			flux_opportunities.append(o.flux(i, j))

plt.loglog(distance, flux_gravity, 'r.')
plt.loglog(distance, flux_radiation, 'b.')
plt.loglog(distance, flux_opportunities, 'g.')