# Field class (relativistic) for OLIVE
#
# Class is initialized with an array of modes and amplitudes as well as corresponding metadata
#
#
# Units
#   -Assume CGS units for now
#

import numpy as np

class Field(object):

    def __init__(self, cavity):
        """Initialize a field object - requires specification of cavity parameters

        Arguments:
            cavity (olive.eigenmodes) - a 'cavity' object with appropriate physical properties

        """

        self.cavity = cavity
        self.modes = []
        self.amplitudes = []
        self.num_modes = 0
        self.frequencies = []
        self.geom_factor = []
        self.poly_exponent = []
        self.w_integrals = []
        self.c_integrals = []
        self.t_integrals = []
        self.x_integrals = []

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


        #Basic mode data
        self.num_modes = np.shape(mode_data)[0]
        self.omegas = np.array(mode_data[:,0])
        self.ks = np.zeros(self.num_modes) #wave numbers

        # Field quantities
        self.ks = ks
        self.omegas = omegas  # these are preloaded as omega/c !
        self.Q = np.asarray(Q0)
        self.P = np.asarray(P0)

        #self.Q = np.zeros(self.num_modes) #self.amplitudes = np.zeros(self.num_modes)
        #self.P = np.zeros(self.num_modes)
        #self.phase_offset = np.zeros(self.num_modes)

        #For rectangular cavities
        #self.M = np.zeros(self.num_modes)
        #self.G = G
        self.M = M #G / (4 * np.pi * c)
        # equations for Ks and Ms for computing field energy
        self.Ml = self.M * np.ones(self.num_modes)
        self.Kl = self.M * (self.modes[:, 0] ** 2 + self.modes[:, 1] ** 2)


        #Mode transverse expansion-unused
        self.geom_factor = np.zeros(self.num_modes)
        self.poly_exponent = np.zeros(self.num_modes)


        #Construct histories
        self.Q_history = [self.Q]
        self.P_history = [self.P]
        #elf.tau_history = [tau0]

        #Mode energy integrals
        #self.w_integrals = np.zeros(self.num_modes)
        #self.c_integrals = np.zeros(self.num_modes)
        #self.t_integrals = np.zeros(self.num_modes)
        #self.x_integrals = np.zeros(self.num_modes)

        #self.wave_vecs = np.array(mode_data[:,1])
        #self.horz_powers = np.array(mode_data[:,2])
        #self.vert_powers = np.array(mode_data[:,3])

        if initial_amplitude:
            if np.shape(mode_data)[0] != np.shape(initial_amplitude):
                msg = 'Number of initial amplitudes must equal the number of modes'
                raise Exception(msg)
            self.Q = np.array(initial_amplitude)

        if initial_modementa:
            if np.shape(mode_data)[0] != np.shape(initial_modementa):
                msg = 'Number of initial amplitudes must equal the number of modes'
                raise Exception(msg)
            self.Q = np.array(initial_amplitude)


        if phase_offset:
            if np.shape(mode_data)[0] != np.shape(phase_offset):
                msg = 'Number of phase offsets must equal the number of modes'
                raise Exception(msg)
            self.phase_offset = np.array(phase_offset)

        if mode_expansion:
            if np.shape(mode_data)[0] != np.shape(mode_expansion)[0]:
                msg = 'Number of mode geometry factors and exponents must equal the number of modes'
                raise Exception(msg)
            self.geom_factor = np.array(mode_expansion[:,0])
            self.poly_exponent = np.array(mode_expansion[:,1])

        if energy_integrals:
            if np.shape(mode_data)[0] != np.shape(energy_integrals)[0]:
                msg = 'Number of sets of energy integrals must equal the number of modes'
                raise Exception(msg)
            if np.shape(energy_integrals)[1] !=4:
                msg = 'Number of integral constants for each mode must equal 4.'
                raise Exception(msg)
            self.w_integrals = np.array(energy_integrals[:,0])
            self.c_integrals = np.array(energy_integrals[:,1])
            self.t_integrals = np.array(energy_integrals[:,2])
            self.x_integrals = np.array(energy_integrals[:,3])


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


    def compute_single_mode_Az(self, index,pos):
        """Compute the z-component of the vector potential Az at position pos for the mode given by index

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
        """Compute the z-component of the vector potential Az for a single position for all modes

        Arguments:
            pos (ndarray): Array of floats specifying position (x,y,z) to compute potential

        Returns:
            Az (float): Value of Az for the specified mode and position

        """

        (tau, x, y) = pos
        expansion_factor = self.geom_factor*(x + 1j*y)**self.poly_exponent

        return expansion_factor*self.amplitudes[index]*np.cos(self.frequencies[index]*tau/c + self.phases[index])


