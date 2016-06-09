
import numpy as np


class field_data:

    def __init__(self):
        self.modes = []
        self.amplitudes = []
        self.num_modes = 0
        self.frequencies = []


    def create_modes(self, mode_data,
                     initial_amplitude=False,
                     initial_phase=False):

        self.num_modes = np.shape(mode_data)[0]
        self.frequencies = np.array(mode_data[:,0])
        self.wave_vecs = np.array(mode_data[:,1])
        self.horz_powers = np.array(mode_data[:,2])
        self.vert_powers = np.array(mode_data[:,3])
        self.amplitudes = np.zeros(self.num_modes)
        self.phases = np.zeros(self.num_modes)

        if initial_amplitude:
            if np.shape(mode_data)[0] != np.shape(initial_amplitude):
                msg = 'Number of initial amplitudes must equal the number of modes'
                raise Exception(msg)
            self.amplitudes = np.array(initial_amplitude)

        if initial_phase:
            if np.shape(mode_data)[0] != np.shape(initial_phase):
                msg = 'Number of initial phases must equal the number of modes'
                raise Exception(msg)
            self.phases = np.array(initial_phase)


    def get_modes(self):

        return self.modes, self.amplitudes, self.phases


    def advance_phases(self, tau):
        """Advance the phases by a time of flight variable"""

        self.phases += tau*self.frequencies