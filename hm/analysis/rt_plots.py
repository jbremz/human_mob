import numpy as np
from matplotlib import pyplot as plt
from hm.analysis.random_tripoint import *

def r_jk_plot(p, model, r_ib, exp = True, tilde = False):
	'''
	Plots epsilon as a function of r_jk, given a costant value of r_ib.
	'''	
	x = mean_r_jk(p, model, r_jk, tilde = False)[0]
	mean_y = mean_r_jk(p, model, r_ib, tilde = False)[1]
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	plt.plot(x*np.sqrt(p.size), mean_y, '.', label = 'Simulation')
	plt.plot(x*np.sqrt(p.size), eps_rjk(p, model, r_ib), '.', label = 'Analytical')
	plt.legend()
	plt.xlabel('$r_{jk} \sqrt{N}$', fontsize = 10)
	plt.ylabel('$\epsilon$', fontsize = 10)
	plt.show()
	
def r_ib_plot(p, model, exp = True, tilde = False):
	'''
	Plots epsilon as a function of r_ib, given a costant value of r_jk.
	'''
	r_jk = 0.06
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	# Resolution
	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi = 500)
		
	
	x, mean_x, mean_y, std = mean_r_ib(p, model, r_jk,exp=exp, tilde = False)
	plt.plot(x*np.sqrt(p.size), eps_rib(p, model, r_jk), label = 'Analytical', color='grey')
	#plt.plot(x*np.sqrt(p.size), mean_y, '.', label = 'Simulation')
	plt.errorbar(np.array(mean_x)*np.sqrt(p.size), mean_y, elinewidth=1, fmt='o', ms=4, yerr=std, label = 'Simulation', color='C5', marker='x')
	plt.xlabel('$r_{ib} \sqrt{N}$', fontsize = 20)
	plt.ylabel('$\epsilon$', fontsize = 20)
	plt.title(r_jk)
	# Legend
	plt.legend(frameon=False, fontsize=20)
	
	# Axes/tick labels
	plt.tick_params(axis='both', labelsize=15)
	plt.ticklabel_format(style='sci')
	
	plt.tight_layout()
	plt.savefig("Radiation 005")

def r_jk_plot(p, model, exp = True, tilde = False):
	'''
	Plots epsilon as a function of r_ib, given a costant value of r_jk.
	'''
	r_jk = 0.06
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	# Resolution
	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi = 500)
		
	
	x, mean_x, mean_y, std = mean_r_ib(p, model, r_jk,exp=exp, tilde = False)
	plt.plot(x*np.sqrt(p.size), eps_rib(p, model, r_jk), label = 'Analytical', color='grey')
	#plt.plot(x*np.sqrt(p.size), mean_y, '.', label = 'Simulation')
	plt.errorbar(np.array(mean_x)*np.sqrt(p.size), mean_y, elinewidth=1, fmt='o', ms=4, yerr=std, label = 'Simulation', color='C5', marker='x')
	plt.xlabel('$r_{ib} \sqrt{N}$', fontsize = 20)
	plt.ylabel('$\epsilon$', fontsize = 20)
	plt.title(r_jk)
	# Legend
	plt.legend(frameon=False, fontsize=20)
	
	# Axes/tick labels
	plt.tick_params(axis='both', labelsize=15)
	plt.ticklabel_format(style='sci')
	
	plt.tight_layout()
	plt.savefig("Radiation 005")
	
### Run this to plot:
	
from hm.pop_models.pop_random import random as pop_random
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation



N = 100
alpha, beta = 1, 1
S = 1/N
# exponential
gamma = 0.3 * (S)**(-0.18)
# power law
#gamma = 1.4 * (S)**(0.11)
p = pop_random(N)
g = gravity(p, alpha, beta, gamma, exp=True)

r = radiation(p)
r_ib_plot(p, r, exp=True)
#r_ib_plot(p, r, 0.1)