#Non-relativistic particle class for OLIVE
#
# Class is initialized with a bunch of particles to provide position and momentum values
# Weightings are optional as the initial implementation is gridless
# 
#

import numpy as np
from scipy.constants import e, m_e, c

class nr_ptcl:
    
    def __init__(self, pdict):
        
        self.num = pdict['num_particles']
        self.dim = pdict['dim']
        self.mass = pdict['mass']
        self.pos = np.zeros((self.num, self.dim))
        self.mom = np.zeros((self.num, self.dim))
        #self.weights = np.zeros(self.num) #not needed for this
        self.exists = np.asarray([False]*self.num)
        
        
    def add_bunch(self, positions, momenta, weights=None):
        '''Initialize bunch of particles. Overwrite position and momentum arrays
        
        Arguments:
            positions (ndarray): array of positions - [x, y, z]
            momenta (ndarray): array of momenta - [px, py, pz]
            weights (Optional[ndarray]): array of weights- [wx,wy,wz]
        
        '''
        
        
        if not type(positions) == 'ndarray' or type(momenta) == 'ndarray':
            positions = np.asarray(positions)
            momenta = np.asarray(momenta)
        
        if not positions.shape(0) == momenta.shape(0):
            print "Position and momentum arrays have unequal lengths"
            raise
        
        self.pos = positions
        self.mom = momenta
        
        
    def compute_energy(self):
        """ 
        Compute the total kinetic energy of each particle

        Returns:
            KE (ndarray): total kinetic energy in ergs
        """

        ke = 0.
        
        #Use numpy Einstein Summation to quickly compute dot products for KE computation
        ke = np.einsum('ij,ij->i',self.mom,self.mom)/(2.*self.mass)

        return ke
        
    def add_particle(self, position, momentum, weight):
        """
        Add a single particle to the bunch.
        
        Arguments:
            positions (ndarray): array of positions - [x, y, z]
            momenta (ndarray): array of momenta - [px, py, pz]
            weights (Optional[ndarray]): array of weights- [wx,wy,wz]
        
        """

        added_ptcl = False
        for idx in range(0, self.num):
            if not self.exists[idx]:
                self.pos[idx,:] = position[:]
                self.mom[idx,:] = momenum[:]
                #self.weights[idx] = weight
                self.exists[idx] = True
                added_ptcl = True
                break

        if not added_ptcl:
            np.append(np.zeros(self.dimension), self.pos)
            np.append(np.zeros(self.dimension), self.mom)
            self.pos[-1,:] = position[:]
            self.mom[-1,:] = momentum[:]
            #self.weights[idx] = weight
            self.exists.append(True)

            added_ptcl = True
            
    def get_particles(self):

        return self.pos, self.vel