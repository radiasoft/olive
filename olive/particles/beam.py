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
from scipy.constants import m_e as me_mks
from scipy.constants import c as c_mks
from olive.dataio import SDDS
from olive.dataio import conversions

# Set the default mass and charge for an electron
m = me_mks*1.e3 #cgs
q = 4.80320451e-10 #esu 1.*e
c = c_mks*1.e2 #cgs

class Beam(object):
    
    def __init__(self, pdict=False):
        '''
        Arguments:
            pdict (Optional[dict]): A dictionary describing species information - keys 'name','mass', 'charge'
        '''

        if pdict:
            #returns false if pdict has no keys
            self.species = pdict['name']
            self.mass = pdict['mass']
            self.charge = pdict['charge']

        else:
            # by default electron
            self.species = 'electron'
            self.mass = me_mks*1e3 #CGS units
            self.charge = q #CGS units

        self.canonical = False #flag determining if momentum is mechanical or canonical
        self.num_particles = 0 #initially empty bunch

        #self.pos = np.zeros((self.num, self.dim))
        #self.mom = np.zeros((self.num, self.dim))
        #self.beta = 0. #single beta for entire bunch!
        #self.weights = np.zeros(self.num) #not needed for this
        #self.exists = np.asarray([False]*self.num)
        
        
    def add_bunch(self, positions, momenta, weights=None, IDs=None):
        '''Initialize bunch of particles. Overwrite position and momentum arrays
        
        Arguments:
            positions (ndarray): array of positions - [x, y, z]
            momenta (ndarray): array of momenta - [px, py, pz]
            weights (Optional[ndarray]): array of weights- [wx,wy,wz]
            IDs (Optional[ndarray]): array of particle IDs - length # of particles
            
        
        '''

        if not type(positions) == 'ndarray':
            positions = np.asarray(positions)
        if not type(momenta) == 'ndarray':
            momenta = np.asarray(momenta)
        if not positions.shape[0] == momenta.shape[0]:
            print "Position and momentum arrays have unequal lengths"
            raise

        # Position quantities
        self.x = positions[:, 0]
        self.y = positions[:, 1]
        self.z = positions[:, 2]

        self.num_particles = len(self.x)

        # initialize particle IDs
        if not IDs is None:
            if len(IDs) == self.num_particles:
                self.IDs = IDs
            else:
                print "Number of particle IDs differs from number of particles"
                raise
        else:
            self.IDs = np.arange(self.num_particles)

        # initialize weights
        if weights is None:
            self.weights = np.ones(self.num_particles)
        elif not type(weights) == 'ndarray':
            weights = np.asarray(weights)
            if len(weights) == self.num_particles:
                self.weights = weights
            else:
                print "Number of particle weights differs from number of particles"
                raise


        # Charge and mass quantities - weighted
        self.mass = self.weights * self.mass
        self.qs = self.weights * self.charge

        # Momentum quantities - weighted
        self.px = self.weights * momenta[:, 0]
        self.py = self.weights * momenta[:, 1]
        self.pz = self.weights * momenta[:, 2]

        #self.tau_history = [tau0]
        #self.pos = positions
        #self.mom = momenta
        #self.beta = self.compute_beta_z()


    def add_from_file(self,file_name):
        '''Add a bunch from an elegant output file. Wraps 'add_bunch'

        Arguments:
            file_name (string): path to elegant output file containing particle data

        '''

        # Instantiate read objects for bunch input and ouput SDDS files
        sdds_file = SDDS.readSDDS(file_name, verbose=False)
        elegant_data = sdds_file.read_columns()
        olive_data = conversions.convert_units_elegant2olive(elegant_data) #convert units

        # Construct bunch from read-in data - data is in the form x,px,y,py,z,pz,ID
        qs = olive_data[:, :6:2]
        ps = olive_data[:, 1:6:2]
        ids = olive_data[:, -1]

        self.add_bunch(qs, ps, IDs=ids)

    def write_to_file(self,file_name,dataMode='binary'):
        '''Write bunch data to SDDS format for elegant interpretation

        Arguments:
            file_name (string): path to elegant output file containing particle data
            dataMode (optional[string]): Mode for writing the file - defaults to binary

        '''

        # Create SDDS output object
        output_file = SDDS.writeSDDS()

        # Convert units back to elegant
        elegant_data = conversions.convert_units_olive2elegant(self.x, self.px, self.y, self.py,
                                                               self.z, self.pz)

        # Columns of data corresponding to necessary attributes
        for i, (dim, unit) in enumerate(zip(('x', 'xp', 'y', 'yp', 't', 'p'), ('m', '', 'm', '', 's', 'm$be$nc'))):
            output_file.create_column(dim, elegant_data[:, i], 'double', colUnits=unit)

        # save file
        output_file.save_sdds(file_name, dataMode=dataMode)



    def convert_mechanical_to_canonical(self,fields):
        '''Convert mechanical momenta to canonical momenta for the current particle state'''

        if not self.canonical:

            A_x = fields.calc_A_x(self.x, self.y, self.z)
            A_y = fields.calc_A_y(self.x, self.y, self.z)
            A_z = fields.calc_A_z(self.x, self.y, self.z)

            self.px = self.px + (self.qs / c) * np.dot(fields.Q, A_x)
            self.py = self.py + (self.qs / c) * np.dot(fields.Q, A_y)
            self.pz = self.pz + (self.qs / c) * np.dot(fields.Q, A_z)

            self.canonical = True

        else:
            print "Momentum is already in canonical form"

    def convert_canonical_to_mechanical(self,fields):
        '''Convert mechanical momenta to canonical momenta for the current particle state'''


        if self.canonical:

            A_x = fields.calc_A_x(self.x, self.y, self.z)
            A_y = fields.calc_A_y(self.x, self.y, self.z)
            A_z = fields.calc_A_z(self.x, self.y, self.z)

            self.px = self.px - (self.qs / c) * np.dot(fields.Q, A_x)
            self.py = self.py - (self.qs / c) * np.dot(fields.Q, A_y)
            self.pz = self.pz - (self.qs / c) * np.dot(fields.Q, A_z)

            self.canonical = False

        else:
            print "Momentum is already in mechanical form"

    def calc_gamma_m_c(self,fields):
        '''Compute the quantity gamma*m*c for every particle and update the corresponding member variable'''

        if self.canonical:

            A_x = fields.calc_A_x(self.x, self.y, self.z)
            A_y = fields.calc_A_y(self.x, self.y, self.z)
            A_z = fields.calc_A_z(self.x, self.y, self.z)

            self.gmc = np.sqrt((self.px - (self.qs / c) * np.dot(fields.Q, A_x)) ** 2 + (
            self.py - (self.qs / c) * np.dot(fields.Q, A_y)) ** 2 + (self.pz - (self.qs / c) * np.dot(fields.Q, A_z)) ** 2 + (
                           self.mass * c) ** 2)

        else:
            self.gmc = np.sqrt(self.px** 2 + self.py** 2 + self.pz** 2 + (self.mass * c) ** 2)

        #self.gmc_history.append(self.gmc / (self.mass * c))
            
    def get_bunch(self):
        """
        Return the 6D phase-space coordinates for the bunch particles in an array.
        
        Returns:
            part_array (ndarray): 6xN array of 6D phase-space coordinates - [x,px,ypy,z,pz]
        
        """
        
        return np.asarray([self.x,self.px,self.y,self.py,self.z,self.pz])

    def total_particle_energy_history(self):
        '''Returns the total particle energy for each timestep in the simulation'''

        return np.dot(self.gmc_history, self.mass) * c * c


    def plot_particle_energy(self):

        particle_energy = self.total_particle_energy_history()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, particle_energy)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Particle Energy [ergs]')
        ax.set_title('Total Particle Energy vs. Time')
        plt.savefig('particle_energy.pdf', bbox_inches='tight')

    def plot_coordinates(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12, 8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, np.asarray(self.z_history)[:, 0], label='z1')
        ax.plot(np.asarray(self.tau_history) / c, np.asarray(self.z_history)[:, 1], label='z2')
        ax.plot(np.asarray(self.tau_history) / c, np.asarray(self.x_history)[:, 0], label='x1')
        ax.plot(np.asarray(self.tau_history) / c, np.asarray(self.x_history)[:, 1], label='x2')
        # ax.plot(np.asarray(self.tau_history)/c,self.y_history[:,0], label = 'y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('Particle in mode 110')
        ax.legend(loc='best')
        plt.savefig('coordinates_110.pdf', bbox_inches='tight')

    def plot_momenta(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12, 8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, np.asarray(self.pz_history)[:, 0] / (m * c ** 2), label='z')
        # ax.plot(np.asarray(self.tau_history)/c,np.asarray(self.px_history)[:,0]/(m*c**2), label = 'x')
        # ax.plot(np.asarray(self.tau_history)/c,self.py_history[:,0], label = 'y')

        # ax.set_ylim([999.5,1000.5])

        ax.set_xlabel('Time')
        ax.set_ylabel('Momenta')
        ax.set_title('Particle in mode 110')
        ax.legend(loc='best')
        plt.savefig('momenta_110.pdf', bbox_inches='tight')