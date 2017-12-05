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

def neighb(p, i):
	'''Returns a list of the nearest neighobours of a given location i as a Numpy array.'''

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
	Returns a list of all nearest neighbours pairs - excluding the target location k -
	as a Numpy array,
	'''

	neighbours = []
	for i in range(p.size):
		if i != k:
			if k not in neighb(p, i):
				neighbours.append(neighb(p, i)[0])

	return np.array(neighbours)

def test_ratio(p, model):
	#delta_theta = 2*np.arctan(r_jk/(2*r_ib))
	#integral = ((2*np.pi- delta_theta)/model.gamma**2) * (model.gamma*np.sqrt(1./p.size))
	r_ij = np.arange(0., 1., 0.01)
	r_jk = 0.1
	r_ib = np.sqrt(r_ij**2 - (r_jk/2.)**2)
	ratio = (np.exp(-model.gamma*r_ij))/(np.exp(-model.gamma*r_ib))
	plt.plot(r_ij, ratio)
	plt.show()

def chi_squared_ib(p, model, r_jk):
chi = []
a_s = []
for a in np.arange(0.5, 3.5, 0.1):
	eps_function = eps_vs_target(p, model, tilde= False)
	x = eps_function[0]
	obs = eps_function[1]
	print(obs)
	pdf_obs = np.bincount(obs)
	print(pdf_obs)
	gamma = model.gamma
	eps_values = 1 - (2*np.exp(-gamma*(np.sqrt(x**2 + (r_jk/2)**2)-x)))/a
	a_s.append(a)
	pdf_exp = np.bincount(ep)
	chi.append(chisquare(obs, f_exp = eps_values))
#return min(chi), a_s[chi.index(min(chi))]