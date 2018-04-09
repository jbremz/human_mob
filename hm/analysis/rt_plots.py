import numpy as np
from matplotlib import pyplot as plt
from hm.analysis.random_tripoint import *
	
def r_plot(p, model, x_value, exp = True, tilde = False):
	'''
	Plots epsilon as a function of r_ib, given a costant value of r_jk.
	'''
	r_jk = 0.06
	
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('seaborn-deep')
	
	# Resolution
	fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi = 500)
		
	if x_value == "r_ib":
		r = 0.06
		x, mean_x, mean_y, std = mean_r(p, model, x_value=x_value, exp=exp, tilde = False)
		plt.plot(x*np.sqrt(p.size), eps_rib(p, model, r), label = 'Analytical', color='grey')
		plt.errorbar(np.array(mean_x)*np.sqrt(p.size), mean_y, elinewidth=1, fmt='o', ms=4, yerr=std, label = 'Simulation', color='C5', marker='x')
		plt.xlabel('$r_{jk} \sqrt{N}$', fontsize = 20)

	if x_value == "r_jk":
		r = 0.4
		#r = np.mean(r_ib(p, 1))
		#print(r)
		#x, mean_x, mean_y, std = mean_r(p, model, x_value=x_value, exp=exp, tilde = False)
		x, mean_x, mean_y, std = rjk_binning(p, model, exp)
		print(std)
		#plt.plot(eps_vs_neighbours(p, model)[0]*np.sqrt(p.size), eps_vs_neighbours(p, model)[1],'o', linewidth=1, ms=3, label = 'Simulation', color='C5', marker='x', zorder = 2)
		plt.errorbar(np.array(mean_x)*np.sqrt(p.size), mean_y, elinewidth=1, fmt='o', ms=4, yerr=np.array(std), label = 'Simulation', color='C5', marker='x')
		plt.plot(x*np.sqrt(p.size), eps_rjk(p, model, r), label = 'Analytical', color='grey', zorder = 1)
		#plt.scatter(np.array(mean_x)*np.sqrt(p.size), mean_y, label = 'Mean', color='red', zorder =3, s=10)
		plt.xlabel('$r_{jk} \sqrt{N}$', fontsize = 20)

	plt.ylabel('$\epsilon$', fontsize = 20)
	plt.title(x_value+" = "+str(r)+" N = "+str(p.size))
	# Legend
	plt.legend(frameon=False, fontsize=20)
	
	# Axes/tick labels
	plt.tick_params(axis='both', labelsize=15)
	plt.ticklabel_format(style='sci')
	
	plt.tight_layout()
	if isinstance(model, gravity):
		mod = "gravity"
	else:
		mod = "radiation"
		
	title = mod+"_exp="+str(exp)+"_"+x_value
	plt.savefig(title)
	
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
g = gravity(p, alpha, beta, gamma, exp=False)

r = radiation(p)
print(np.mean(r_jk(p, 1)))

r_plot(p, r, "r_jk",  exp=False)
#r_ib_plot(p, r, 0.1)