# -*- coding: utf-8 -*-
u"""Interface class for creating and running simulations using OLIVE
:copyright: Copyright (c) 2016 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""

import numpy as np
from scipy.constants import c as c_mks
import matplotlib as mpl
import matplotlib.pyplot as plt

c = c_mks*1.e2

class simulator(object):

    def __init__(self,beam,fields,maps):
        '''Initializes a simulator with beam, field, and map objects'''

        self.beam = beam
        self.fields = fields
        self.maps = maps


    def define_simulation(self, maxTau, numSteps):
        '''Provide times and timestep information for simulation. Defines history arrays.

        '''

        self.maxTau = maxTau
        self.numSteps = numSteps
        self.h = maxTau/numSteps
        self.tau = 0
        self.tau_history = [self.tau]

        self.beam.convert_mechanical_to_canonical(self.fields)
        self.beam.calc_gamma_m_c(self.fields)
        self.gmc_history = [self.beam.gmc]

        # Construct history arrays to store updates
        self.x_history = [self.beam.x]
        self.y_history = [self.beam.y]
        self.z_history = [self.beam.z]
        self.px_history = [self.beam.px]
        self.py_history = [self.beam.py]
        self.pz_history = [self.beam.pz]

        self.Q_history = [self.fields.Q]
        self.P_history = [self.fields.P]


    def update_histories(self):
        '''Update coordinate histories with new values'''

        self.x_history.append(self.beam.x)
        self.y_history.append(self.beam.y)
        self.z_history.append(self.beam.z)
        self.px_history.append(self.beam.px)
        self.py_history.append(self.beam.py)
        self.pz_history.append(self.beam.pz)

        self.Q_history.append(self.fields.Q)
        self.P_history.append(self.fields.P)


    def step(self, N=1):
        '''Perform N steps in c-tau'''

        self.maps.initialize_beam(self.beam, self.fields)

        h = self.h

        print "Beginning step 1"

        i = N - 0.5

        while i > 0:
            self.beam.calc_gamma_m_c(self.fields)
            self.gmc_history.append(self.beam.gmc)
            self.maps.rotate_fields(self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_z(self.beam, self.fields, h, step=1.0)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.rotate_fields(self.fields, h, step=0.5)

            # update times for diagnostics
            self.tau = self.tau + h
            self.tau_history.append(self.tau)

            # update coordinate histories
            self.update_histories()

            i = i - 1

    ####################################
    ## History computations and getters
    ####################################

    def mode_energy_history(self):
        '''Returns an array of mode-energies for all timesteps in the simulation'''

        bQH = np.asarray(self.Q_history)  # convert to numpy array
        bPH = np.asarray(self.P_history)  # convert to numpy array

        E_modes = c * (0.5 * bPH * bPH / self.fields.Ml + 0.5 * self.fields.Kl * bQH * bQH)

        return E_modes

    def total_particle_energy_history(self):
        '''Returns the total particle energy for each timestep in the simulation'''

        return np.dot(self.gmc_history, self.beam.mass) * c * c


    def energy_change_sytem(self):
        '''Return the change in total system energy over the course of the simulation'''

        Ef_total = np.sum(self.mode_energy_history(), 1)  # get the mode energies and sum over them for each timesetp
        Ep_total = self.total_particle_energy_history()

        E_tot = Ef_total + Ep_total

        return ((E_tot[-1] - E_tot[0]) / (E_tot[0])) * 100.

    ####################################
    ## Plotting commands for diagnostics
    ####################################


    def plot_total_system_energy(self):
        '''Compute and plot the total energy - particles plus fields - for the simulation'''

        Ef_total = np.sum(self.mode_energy_history(), 1)  # get the mode energies and sum over them for each timesetp
        Ep_total = self.total_particle_energy_history()

        E_tot = Ef_total + Ep_total

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, E_tot)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Total Energy [ergs]')
        ax.set_title('Total System Energy vs. Time')
        plt.savefig('total_energy.pdf', bbox_inches='tight')

    def plot_particle_energy(self):

        particle_energy = self.total_particle_energy_history()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, particle_energy)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Particle Energy [ergs]')
        ax.set_title('Total Particle Energy vs. Time')
        plt.savefig('particle_energy.pdf', bbox_inches='tight')


    def plot_mode_energies(self):
        E_modes = self.mode_energy_history()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()

        for index, single_mode_E in enumerate(E_modes.T):
            ax.plot(np.asarray(self.tau_history) / c, single_mode_E, label='Mode {}'.format(index))

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Field Energy [ergs]')
        ax.set_title('Mode Energy vs. Time')
        plt.savefig('mode_energy.pdf', bbox_inches='tight')


    def plot_field_rotations(self):

        fig = plt.figure(figsize=(12, 8))

        ax = fig.gca()

        Q0s = np.asarray([Q[0] for Q in self.Q_history])
        # rescale P by 1/M*omega to normalize
        scaled_P0s = np.asarray([P[0] / (self.fields.Ml[0] * self.fields.omegas[0]) for P in self.P_history])

        ax.plot(Q0s, scaled_P0s)

        ax.set_xlabel(r'Q')
        ax.set_ylabel(r'P/$M \Omega$')
        ax.set_title('Fields in mode 110')
        plt.savefig('field_rotation.pdf', bbox_inches='tight')


    def plot_field_amplitudes(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12, 8))

        ax = fig.gca()

        new_Qs = np.asarray([Q[0] for Q in self.Q_history])

        ax.plot(np.asarray(self.tau_history) / c, new_Qs, label=r'$a_0$')


        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Fields in mode 110')
        ax.legend(loc='best')
        plt.savefig('amplitudes_110.pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        new_Ps = np.asarray([P[0] for P in self.P_history])
        leg_Ps = np.concatenate(([new_Ps[0]], new_Ps[::2]))
        ax.plot(np.asarray(self.tau_history) / c, leg_Ps, label=r'$a_0$')

        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Fields in mode 110')
        ax.legend(loc='best')
        plt.savefig('oscillations_110.pdf', bbox_inches='tight')
