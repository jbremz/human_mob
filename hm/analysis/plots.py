from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.pop_models.pop_random import random as pop_random
from matplotlib import pyplot as plt
import numpy as np

def plot_flow(population, model='all', alpha = 1, beta = 1, gamma = 0.2):

	'''
	Takes a population object and a mobility model and plots the flow probability
	as a function of the (scaled) distance between two locations

	TODO: pass instance of mob_model as second argument
	'''


	distance = []
	flux_gravity = []
	flux_radiation = []
	flux_opportunities = []
	p = population
	g = gravity(p, alpha, beta, gamma)
	r = radiation(p)
	o = opportunities(p, gamma)

	for i in range(p.size):
		for j in range(p.size):
			if i != j:
				distance.append(p.r(i, j)*np.sqrt(p.size))

				if model == 'all':
					flux_gravity.append(g.flux(i, j))
					flux_radiation.append(r.flux(i, j))
					flux_opportunities.append(o.flux(i, j))

				if isinstance(model, gravity) == True:
					flux_gravity.append(g.flux(i, j))

				if isinstance(model, radiation) == True:
					flux_radiation.append(r.flux(i, j))

				if isinstance(model, opportunities) == True:
					flux_opportunities.append(o.flux(i, j))

	plt.loglog(distance, flux_gravity, '.', label = 'gravity')
	#plt.loglog(distance, flux_radiation, '.', label = 'radiation')
	#plt.loglog(distance, flux_opportunities, '.', label = 'opportunities')

	plt.xlabel(r'$ \~r$')
	plt.ylabel('$p_{ij}$')
	plt.legend()
	plt.title('Log-log plot')
	plt.show()