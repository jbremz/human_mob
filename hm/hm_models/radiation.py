from hm.utils.utils import disp
import numpy as np
from .base_model import mob_model

class radiation(mob_model):
	'''
	The normalised radiation human mobility model
	'''
	def flux(self, i, j, probs=True):
		'''
		Takes the indices of two locations i, j and returns the average flux from i to j
		'''
		pop = self.pop
		popi, popj = pop.popDist[i], pop.popDist[j]
		popSij = pop.s(i, j)

		# Probabilities
		if probs:
			n = (1/(1-popi/pop.M))*(popi*popj)/float((popi+popSij)*(popi+popSij+popj))

		# Flows
		if not probs:
			n = (popi/(1-popi/pop.M))*(popi*popj)/float((popi+popSij)*(popi+popSij+popj))

		return n