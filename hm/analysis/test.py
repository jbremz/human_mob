import importlib
from hm.hm_models.gravity import gravity
from hm.hm_models.radiation import radiation
from hm.hm_models.opportunities import opportunities
from hm.pop_models.pop_random import random as pop_random
from hm.pop_models.pop_explicit import explicit as pop_explicit

popDist = [3,4,7,5,6]
locCoords = [[2,3],[3,2],[-5,9],[0,1],[1,-8]]
alpha, beta = 1, 1
gamma = 0.2
N = 20

p = pop_random(N)
p1 = pop_explicit(locCoords, popDist)
g = gravity(p, alpha, beta, gamma)
r = radiation(p)
o = opportunities(p, gamma) # TODO seems a little slow, probably just the nature of the algorithm