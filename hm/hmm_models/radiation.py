from utils import disp
import numpy as np
from .base_model import mob_model

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