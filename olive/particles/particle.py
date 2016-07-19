# Particle class (relativistic) for OLIVE
#
# Class is initialized with a bunch of particles to provide position and momentum values
# Weightings are optional as the initial implementation is gridless
# 
# Keep in mind that z is the independent variable, and thus x,y, and tau = c*t are the coordinate
# descriptors of the particles, along with px, py, and ptau.
#
# Coordinates:
#   - Particles have position x, y, tau   
#
# Momenta
#   - Particles have momenta p_x p_y, p_tau
#   - We assume a fixed beta for the bunch, computed by averaging over all tau velocities
# 
# Units
#   -Assume CGS units for now
#

import numpy as np
from scipy.constants import e, m_e, c

class Particle(object):
    
    def __init__(self, pdict):
        
        self.num = pdict['num_p']
        self.dim = pdict['dim']
        self.mass = pdict['mass']
        self.pos = np.zeros((self.num, self.dim))
        self.mom = np.zeros((self.num, self.dim))
        self.beta = 0. #single beta for entire bunch!
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
        
        if not positions.shape[0] == momenta.shape[0]:
            print "Position and momentum arrays have unequal lengths"
            raise
        
        self.pos = positions
        self.mom = momenta
        self.beta = self.compute_beta_z()
        
        
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
        
    def compute_gamma(self):
        """
        Compute the relativistic gamma for each particle
        
        Returns:
            gamma (ndarray): relativistic gamma - unitless
        
        """

        ptau_avg = np.mean(np.einsum('ij,ij->i',self.mom,self.mom))
        return np.sqrt((ptau_avg*c)**2 + (self.mass*c**2)**2)

    def compute_gamma_z(self):
        """
        Compute the relativistic gamma for motion in the z/tau axis for each particle

        Returns:
            gamma (ndarray): relativistic gamma - unitless

        """

        ptau_avg = np.mean(self.mom[:,2])
        return np.sqrt((ptau_avg*c)**2 + (self.mass*c**2)**2)/(self.mass*c**2)
        
        
    def compute_beta_z(self):
        """
        Compute the beta-z value for the bunch. We assume a single velocity by averaging over
        all values of p_tau in the bunch.

        Returns:
            beta_z(ndarray): relativistic beta along the longitudinal axis - unitless
        
        """
        
        #ptau_avg = np.mean(self.mom[:,2])
        #energy_tau_avg = np.sqrt((ptau_avg*c)**2 + (self.mass*c**2)**2)
        #gamma_tau_avg = energy_tau_avg/(self.mass*c**2)

        gamma_tau_avg = self.compute_gamma_z()
        #ke_tau_avg = ptau_avg*ptau_avg/(2*self.mass)
        beta_z_avg = np.sqrt(1-1./gamma_tau_avg**2)
        
        return beta_z_avg
        
    def add_particle(self, position, momentum, weight):
        """
        Add a single particle to the bunch.
        
        Arguments:
            positions (ndarray): array of positions - [x, y, tau]
            momenta (ndarray): array of momenta - [px, py, ptau]
            weights (Optional[ndarray]): array of weights- [wx,wy,wtau]
        
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
            np.append(np.zeros(self.dim), self.pos)
            np.append(np.zeros(self.dim), self.mom)
            self.pos[-1,:] = position[:]
            self.mom[-1,:] = momentum[:]
            #self.weights[idx] = weight
            self.exists.append(True)

            added_ptcl = True
            
        #now update beta
        self.beta = self.compute_beta_z()
            
    def get_particles(self):
        """
        Return the 6D phase-space coordinates for the bunch particles in an array.
        
        Returns:
            part_array (ndarray): array of 6D phase-space coordinates - [x,y,tau,px,py,ptau]
        
        """
        
        return np.hstack([self.pos,self.mom])