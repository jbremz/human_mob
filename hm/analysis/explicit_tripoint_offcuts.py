
def epsChangeX(xmin, xmax, y, n, N, model='gravity', ib=False, analytical=False, gamma=2):
	'''
	Fixes y and varies x across n values between ymin and ymax for a random distribution of N locations

	'''
	x = np.linspace(xmin, xmax, n)

	epsVals = []

	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	if model=='gravity':
		func = epsilon_g
	if model=='radiation':
		func = epsilon_r
	if model=='opportunities':
		func = epsilon_io

	for val in x:
		epsVals.append(abs(func(val, y, N, ib=ib, seed=seed)))

	xEps = np.array([x * np.sqrt(N), np.array(epsVals)]).T

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.scatter(xEps[:,0], xEps[:,1], s=10, label='Simulation')

	if analytical:
		anlytXEps = np.array([x * np.sqrt(N), anlyt_epsilon(x, y, gamma=gamma)]).T
		ax.scatter(anlytXEps[:,0], anlytXEps[:,1], s=10, label='Analytical Result')

	ax.legend()

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{ib} \sqrt{N}$')
	ax.set_ylabel(r'$\epsilon$')

	plt.show()

	return

def epsChangeY(ymin, ymax, x, n, N, model='gravity', ib=False, analytical=False, gamma=20):
	'''
	Fixes x and varies y across n values between ymin and ymax for a random distribution of N locations

	'''
	y = np.linspace(ymin, ymax, n)
	epsVals = []
	seed = int(np.random.rand(1)[0] * 10000000) # so that all the random population distriubtions are the same

	if model=='gravity':
		func = epsilon_g
	if model=='radiation':
		func = epsilon_r
	if model=='opportunities':
		func = epsilon_io

	for val in y:
		epsVals.append(abs(func(x, val, N, ib=ib, seed=seed)))

	yEps = np.array([y * np.sqrt(N), np.array(epsVals)]).T

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.scatter(yEps[:,0], yEps[:,1], s=10, label='Simulation')

	if analytical:
		anlytYEps = np.array([y * np.sqrt(N), anlyt_epsilon(x, y, gamma=gamma)]).T
		ax.scatter(anlytYEps[:,0], anlytYEps[:,1], s=10, label='Analytical Result')

	ax.legend()

	plt.rc('text', usetex=True)

	ax.set_xlabel(r'$r_{jk} \sqrt{N}$')
	ax.set_ylabel(r'$\epsilon$')

	plt.show()

	return