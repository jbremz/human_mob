
from utils import disp
import numpy as np

class pop_distribution:
	def __init__(self, popDist, locCoords):
		self.locCoords = locCoords
		self.popDist = popDist
		self.size = len(self.locCoords)
		
	def s(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the population in a circle of radius r 
		(= distance between two locations) centred on i 
		'''
		r = disp(self.locCoords[i], self.locCoords[j])
		closer_pop = []
		for loc in range(self.size):
			if disp(self.locCoords[i], self.locCoords[loc]) <= r:
				if loc != i:
					if loc != j:
						closer_pop.append(self.popDist[loc])
						
		return sum(closer_pop)

	def M(self):
		'''
		Returns the total sample population

		'''
		return np.sum(np.array(self.popDist))

class mob_model:
	'''
	Base human mobility model class
	'''        
	def __init__(self, pop):
		self.pop = pop # the population object

	def ODM(self):
		'''
		returns the predicted origin-destination flow matrix for the population distribution

		'''
		pop = self.pop
		e = len(pop.locCoords)
		m = np.zeros((pop.size, pop.size)) # original OD matrix to be filled with fluxes

		for i in range(pop.size):
			for j in range(i+1,pop.size):
				f = self.flux(i,j)
				m[i][j], m[j][i] = f, f # symmetrical
				
		return m


class gravity(mob_model):
	'''
	The gravity human mobility model

	'''

	def __init__(self, pop, alpha, beta, gamma, K, **kwargs):
		super().__init__(pop)
		kwargs.setdefault('exp', False)
		self.alpha = alpha # population i exponent
		self.beta = beta # population j exponent
		self.gamma = gamma # distance exponent
		self.K = K # fitting parameter
		self.exp = kwargs['exp'] # True if exponential decay function is used, False if power decay is used

	def f(self, r):
		'''
		The distance decay function
		'''
		if self.exp:
			return np.exp(-self.gamma*r)
		else:
			return r**(-self.gamma)
		
	def flux(self, i, j):
		'''
		Takes the indices of two locations and returns the flux between them
		'''

		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		r = disp(pop.locCoords[i], pop.locCoords[j])
		n = self.K * ((popi**self.alpha)*(popj**self.beta))*self.f(r)
		
		return n 
	
class radiation(mob_model):
	'''
	The normalised radiation human mobility model
	'''   
	def flux(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the average flux from i to j
		'''
		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		popSij = pop.s(i, j)
		n = (popi/(1-popi/pop.M()))*(popi*popj)/float((popi+popSij)*(popi+popSij+popj)) # TODO how do we define Ti here?
		
		return n
	
class opportunities(mob_model):
	''' 
	The intervening opportunities model
	'''
	def __init__(self, pop, gamma):
		super().__init__(pop)
		self.gamma = gamma 
		
	def norm_factor(self, i, j):
		pop = self.pop
		to_sum = []
		for k in range(pop.size):
			if k != j:
				a = np.exp((-self.gamma*pop.s(i, k)))-(np.exp(-self.gamma*(pop.s(i, k)+pop.Dist[k])))
				to_sum.append(a)
		return sum(to_sum)
	
	def flux(self, i, j):
		'''
		Takes the indices of two locations i, j and returns the average flux from i to j
		'''
		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		popSij = pop.s(i, j)
		a = np.exp((-self.gamma*popSij)-(np.exp(-self.gamma*(popSij)+pop.Dist[j])))
		n = self.norm_factor(i, j)*popi*a
		return n
		
# Test Data




popDist = [3,5,6,2,1]
locCoords = np.array([[0,0],[1,3],[4,7],[-3,-5],[6,-9]])
beta = 0.2  
K = 10







