from hm.hmm_models.gravity import gravity
from hm.hmm_models.radiation import radiation
from hm.hmm_models.opportunities import opportunities
from hm.pop_models.population import pop_distribution as pop_dist

popDist = [3,4,7,5,6]
locCoords = [[2,3],[3,2],[-5,9],[0,1],[1,-8]]
alpha, beta, K = 1, 1, 1
gamma = 0.2

p = pop_dist(popDist, locCoords)
g = gravity(p, alpha, beta, gamma,K)
r = radiation(p)
o = opportunities(p, gamma)