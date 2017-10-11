
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
    
    
class mob_model:
    '''
	Base human mobility model class
    '''        
    def __init__(self, pop):
        self.pop = pop
        
    def ODM(self):
        '''
        urns the predicted origin-destination flow matrix for the population distribution

        '''
        e = len(self.locCoords)
        np.zeros((size, size)) # original OD matrix to be filled with fluxes

        for i in range(size):
            for j in range(i+1,size):
                f = self.flux(i,j)
                print([i,j], f)
                m[i][j], m[j][i] = f, f # symmetrical
                
	    return m


class simple_gravity(mob_model):
    '''
    The simple gravity human mobility model

    '''
    
    def __init__(self, popDist, locCoords, beta, K):
	self.beta = beta # inverse distance exponent
	self.K = K # fitting parameter
        
    def flux(self, i, j):
        '''
        Takes the indices of two locations and returns the flux between them
        '''
        
        pop = self.pop
        popi, popj = pop.popDist[i], pop.popDist[j]
        r = disp(pop.locCoords[i], pop.locCoords[j])
        self.K * (popi*popj)/r**self.beta
        
        return n 
    
class radiation(mob_model):
    '''
    The radiation human mobility model
    '''   
    def flux(self, i, j):
        '''
        Takes the indices of two locations i, j and returns the average flux from i to j
        '''
        pop = self.pop
        popi, popj = pop.popDist[i], pop.popDist[j]
        O_i = popi
        n = popi*(popi*popj)/float((popi+pop.s(i, j))*(popi+pop.s(i, j)+popj))
        
        return n