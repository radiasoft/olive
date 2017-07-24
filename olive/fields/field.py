# Field class (relativistic) for OLIVE
#
# Class is initialized with an array of modes and amplitudes as well as corresponding metadata
#
#
# Units
#   -Assume CGS units for now
#

import numpy as np
from scipy.constants import c as c_mks

c = c_mks*1.e2

class Field(object):

    def __init__(self, cavity):
        """Initialize a field object - need to specify a cavity

        Arguments:
            cavity (olive.fields.eigenmodes.<ModeSubClass>): an Olive object containing geometry and mode information

        """
        self.cavity = cavity
        self.modes = []
        self.Q = []
        self.P = []
        self.num_modes = 0
        #self.frequencies = []
        self.geom_factor = []
        self.poly_exponent = []
        self.w_integrals = []
        self.c_integrals = []
        self.t_integrals = []
        self.x_integrals = []
        self.det = []

    def create_modes(self,mode_nums, initial_amplitude, initial_modementa, phase_offset=False,
                     mode_expansion=False, energy_integrals=False):

        """Fills in corresponding field data using input arrays containing mode data for L modes. Because we
        pass a cavity argument to the class we can get most of the static information from that.

        Arguments:
            mode_nums(ndarray): Lx3 array containing mode numbers m,n,p.
            initial_amplitude(ndarray): Vector of length num_modes containing initial mode amplitudes
            initial_modementa(ndarray): Vector of length num_modes containing initial modementa
            mode_data(ndarray): 2xL array containing L mode frequencies and L mode wavenumbers
            phase_offset(ndarray): Vector of length num_modes containing phase offset information
            mode_expansion(ndarray): 2 X num_modes array containing geometry factors and exponents for each mode
            energy_integrals(ndarray): 4 X num_modes array containing integral constants for each each mode

        """

        # Basic mode data
        self.num_modes = np.shape(mode_nums)[0]

        if len(initial_amplitude) > 0:
            if  not np.shape(initial_amplitude)[0] == self.num_modes:
                msg = 'Number of initial amplitudes must equal the number of modes'
                raise Exception(msg)
            self.Q = np.asarray(initial_amplitude)
        else:
            msg = 'Must specify positive number of initial amplitudes Q0'
            raise Exception(msg)

        if len(initial_modementa) > 0:
            if not np.shape(initial_modementa)[0] == self.num_modes:
                msg = 'Number of initial amplitudes must equal the number of modes'
                raise Exception(msg)
            self.P = np.asarray(initial_modementa)
        else:
            msg = 'Must specify positive number of initial modementa P0'
            raise Exception(msg)

        # Mode frequencies and wavenumbers
        self.omegas = self.cavity.get_mode_frequencies(mode_nums[:, 0], mode_nums[:, 1],
                                                       mode_nums[:, 2]) / c  # omega over c
        self.ks = self.cavity.get_mode_wavenumbers(mode_nums[:, 0], mode_nums[:, 1], mode_nums[:, 2])  # wave numbers
        self.kx = self.ks[0]
        self.ky = self.ks[1]
        self.kz = self.ks[2]

        # Field geometry quantities
        self.M = self.cavity.M
        self.Ml = self.cavity.get_mode_Ms(self.kx, self.ky, self.kz)
        self.Kl = self.Ml * (self.kx ** 2 + self.ky ** 2)




        #Construct histories
        self.Q_history = [self.Q]
        self.P_history = [self.P]



        #DEPRECATED INITIALIZATIONS FOR TRANSVERSE MODE EXPANSION

        #Mode transverse expansion
        #self.geom_factor = np.zeros(self.num_modes)
        #self.poly_exponent = np.zeros(self.num_modes)

        #Mode energy integrals
        #self.w_integrals = np.zeros(self.num_modes)
        #self.c_integrals = np.zeros(self.num_modes)
        #self.t_integrals = np.zeros(self.num_modes)
        #self.x_integrals = np.zeros(self.num_modes)

        #self.wave_vecs = np.array(mode_data[:,1])
        #self.horz_powers = np.array(mode_data[:,2])
        #self.vert_powers = np.array(mode_data[:,3])


    def add_mode(self,frequency, initial_amplitude=False,
                     initial_phase=False, mode_expansion=False,
                     energy_integrals=False):
        '''Add a single mode to the current Field object

        Arguments:
            frequency (float): Mode frequency
            initial_amplitude (float): Initial mode amplitude
            phase_offset (float): Phase offset for mode
            mode_expansion (ndarray): Pair containing containing geometry factor and exponent for mode
            energy_integrals (ndarray): Quadruplet containing integral constants for  mode

        '''


    def return_modes(self):
        '''Return the mode frequencies and amplitudes'''

        return self.modes, self.amplitudes

    def calc_A_x(self, x, y, z):
        '''
        Returns an LxN array of A_x for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            A_x (ndArray): An LxN array of values
        '''

        return self.cavity.calc_A_x(self.kx, self.ky, self.kz, x, y, z)

    def calc_A_y(self, x, y, z):
        '''
        Returns an LxN array of A_y for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            A_y (ndArray): An LxN array of values
        '''

        return self.cavity.calc_A_y(self.kx, self.ky, self.kz, x, y, z)

    def calc_A_z(self, x, y, z):
        '''
        Returns an LxN array of A_x for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            A_z (ndArray): An LxN array of values
        '''

        return self.cavity.calc_A_z(self.kx, self.ky, self.kz, x, y, z)


    def dx_int_A_z(self, x, y, z):
        '''
        Returns an LxN array of x derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dx_int_A_z (ndArray): An LxN array of values
        '''

        return self.cavity.dx_int_A_z(self.ks, x, y, z)

    def dy_int_A_z(self, x, y, z):
        '''
        Returns an LxN array of y derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dy_int_A_z (ndArray): An LxN array of values
        '''

        return self.cavity.dy_int_A_z(self.ks, x, y, z)

    def dz_int_A_z(self, x, y, z):
        '''
        Returns an LxN array of z derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dz_int_A_z (ndArray): An LxN array of values
        '''

        return self.cavity.dz_int_A_z(self.ks, x, y, z)

    def calc_int_A_z(self, x, y, z):

        '''
        Returns an LxN array of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            int_A_z (ndArray): An LxN array of values
        '''

        return self.cavity.calc_int_A_z(self.ks,x,y,z)


    def compute_single_mode_Az(self, index,pos):
        """DEPRECATED - Compute the z-component of the vector potential Az at position pos for the mode given by index

        Arguments:
            index (int): Index of the desired mode
            pos (ndarray): Array of floats specifying position (x,y,z) to compute potential

        Returns:
            Az (float): Value of Az for the specified mode and position

        """

        (tau, x, y) = pos
        expansion_factor = self.geom_factor[index]*(x + 1j*y)**self.poly_exponent

        return expansion_factor*self.amplitudes[index]*np.cos(self.frequencies[index]*tau/c + self.phases[index])

    def compute_all_modes_Az(self,pos):
        """DEPRECATED - Compute the z-component of the vector potential Az for a single position for all modes

        Arguments:
            pos (ndarray): Array of floats specifying position (x,y,z) to compute potential

        Returns:
            Az (float): Value of Az for the specified mode and position

        """

        (tau, x, y) = pos
        expansion_factor = self.geom_factor*(x + 1j*y)**self.poly_exponent

        return expansion_factor*self.amplitudes[index]*np.cos(self.frequencies[index]*tau/c + self.phases[index])


