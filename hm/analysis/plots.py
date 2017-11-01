from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.pop_models.pop_random import random as pop_random
from matplotlib import pyplot as plt
import numpy as np

alpha, beta = 1, 1
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
			distance.append(p.r(i, j)/np.sqrt(N))
			flux_gravity.append(g.flux(i, j))
			flux_radiation.append(r.flux(i, j))
			flux_opportunities.append(o.flux(i, j))

plt.loglog(distance, flux_gravity, '.',markersize = 3, label = 'Gravity')
plt.loglog(distance, flux_radiation, '.', markersize = 3,label = 'Radiation')
plt.loglog(distance, flux_opportunities, '.', markersize = 3, label = 'Opportunities')

plt.xlabel(r'$ \frac{r}{\sqrt{N}}$')
plt.ylabel('$p_{ij}$')
plt.legend()
plt.title('Log-log plot')