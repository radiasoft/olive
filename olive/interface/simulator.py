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
        '''Initializes a simulator with beam, field, and map objects

         Arguments:
            beam (olive.particles.beam.Beam): Olive object containing beam information
            fields (olive.fields.field.Field): Olive object containing field information
            maps (olive.maps.<Map sub-class>.Map): Olive object containing map information
        '''

        self.beam = beam
        self.fields = fields
        self.maps = maps


    def define_simulation(self, maxTau, numSteps):
        '''Provide times and timestep information for simulation. Defines history arrays.

        Arguments:
            maxTau (float): final z-value of the simulation
            numSteps (int): number of steps for the simulation

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


    def step_with_determinant(self, N=1, diagPeriod=1):
        '''Perform N steps in c-tau and compute determinant of rotation matrix as well.

        Arguments:
            N (Optional[int]): Number of steps to perform. Defaults to 1.
            diagPeriod (Optional[int]): Number of steps to perform in between history updates

        '''

        self.maps.initialize_beam(self.beam, self.fields)

        h = self.h
        diagCount = 0
        diagTurn = False

        print "Beginning step 1"

        i = N - 0.5

        while i > 0:
            if diagCount%diagPeriod == 0 or i < 1:
                diagTurn = True

            self.maps.rotate_fields_determinant(self.fields, h, step=0.5)
            self.beam.calc_gamma_m_c(self.fields)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_z(self.beam, self.fields, h, step=1.0)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.rotate_fields_determinant(self.fields, h, step=0.5)

            # update times for diagnostics
            self.tau = self.tau + h

            #update histories
            if diagTurn:
                self.gmc_history.append(self.beam.gmc)
                self.tau_history.append(self.tau)
                self.update_histories()  # update coordinate histories

            i = i - 1
            diagCount = diagCount+1
            diagTurn = False



    def step(self, N=1, diagPeriod=1):
        '''Perform N steps in c-tau

        Arguments:
            N (Optional[int]): Number of steps to perform. Defaults to 1.
            diagPeriod (Optional[int]): Number of steps to perform in between history updates

        '''

        self.maps.initialize_beam(self.beam, self.fields)

        h = self.h
        diagCount = 0
        diagTurn = False

        print "Beginning step 1"

        i = N - 0.5

        while i > 0:
            if diagCount%diagPeriod == 0 or i < 1:
                diagTurn = True

            self.maps.rotate_fields(self.fields, h, step=0.5)
            self.beam.calc_gamma_m_c(self.fields)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_z(self.beam, self.fields, h, step=1.0)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.rotate_fields(self.fields, h, step=0.5)

            # update times for diagnostics
            self.tau = self.tau + h

            #update histories
            if diagTurn:
                self.gmc_history.append(self.beam.gmc)
                self.tau_history.append(self.tau)
                self.update_histories()  # update coordinate histories

            i = i - 1
            diagCount = diagCount+1
            diagTurn = False


    ####################################
    ## History computations and getters
    ####################################

    def mode_energy_history(self):
        '''Returns an array of mode-energies for all timesteps in the simulation

        Returns:
            E_modes (ndarray): numModes x numSteps array of mode energies

        '''

        bQH = np.asarray(self.Q_history)  # convert to numpy array
        bPH = np.asarray(self.P_history)  # convert to numpy array

        E_modes = c * (0.5 * bPH * bPH / self.fields.Ml + 0.5 * self.fields.Kl * bQH * bQH)

        return E_modes

    def total_particle_energy_history(self):
        '''
        Return the total particle energy for each step in the simulation

        Returns:
            energy_array (ndarray): 1 x numSteps array of particle energy in ergs

        '''
        return np.dot(self.gmc_history, self.beam.mass) * c * c

    def energy_change_sytem_ergs(self):
        '''Return the change in total system energy over the course of the simulation in ergs

        '''

        #grab initial and final particle energy values
        p_e_i = self.gmc_history[0]
        p_e_f = self.gmc_history[-1]
        Ep_change = np.dot((p_e_f-p_e_i), self.beam.mass)* c * c

        # get the mode energies and sum over them for each timestep
        Ef_total = np.sum(self.mode_energy_history(), 1)
        Ef_change = Ef_total[-1]-Ef_total[0]

        return Ep_change + Ef_change

    def energy_change_sytem(self):
        '''Return the change in total system energy over the course of the simulation as a % of the total energy

        '''
        #grab initial and final particle energy values
        p_e_i = self.gmc_history[0]
        p_e_f = self.gmc_history[-1]
        Ep_final = np.dot(p_e_f, self.beam.mass)* c * c
        Ep_change = np.dot((p_e_f-p_e_i), self.beam.mass)* c * c


        # get the mode energies and sum over them for each timestep
        Ef_total = np.sum(self.mode_energy_history(), 1)
        Ef_change = Ef_total[-1]-Ef_total[0]

        E_final = Ep_final + Ef_total[-1]

        return 100.*(Ep_change + Ef_change) /E_final

    ####################################
    ## Plotting commands for diagnostics
    ####################################


    def plot_total_system_energy(self, save=True):
        '''Compute and plot the total energy - particles plus fields - for the simulation

        Arguments:
            save (Optional[bool]): If true, save the figure. Defaults to True

        '''

        Ef_total = np.sum(self.mode_energy_history(), 1)  # get the mode energies and sum over them for each timestep
        Ep_total = self.total_particle_energy_history()

        E_tot = Ef_total + Ep_total

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, E_tot)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Total Energy [ergs]')
        ax.set_title('Total System Energy vs. Time')
        if save:
            plt.savefig('total_energy.pdf', bbox_inches='tight')

    def plot_particle_energy(self,save=True):
        '''
        Plot the total particle energy through the simulation history

        Arguments:
            save (Optional[bool]): If true, save the figure. Defaults to True
        '''

        particle_energy = self.total_particle_energy_history()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history) / c, particle_energy)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Particle Energy [ergs]')
        ax.set_title('Total Particle Energy vs. Time')
        if save:
            plt.savefig('particle_energy.pdf', bbox_inches='tight')


    def plot_mode_energies(self, save=True):
        '''Compute and plot the individual mode energies for the simulation

        Arguments:
            save (Optional[bool]): If true, save the figure. Defaults to True

        '''

        E_modes = self.mode_energy_history()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()

        for index, single_mode_E in enumerate(E_modes.T):
            ax.plot(np.asarray(self.tau_history) / c, single_mode_E, label='Mode {}'.format(index))

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Field Energy [ergs]')
        ax.set_title('Mode Energy vs. Time')
        if save:
            plt.savefig('mode_energy.pdf', bbox_inches='tight')


    def plot_field_rotations(self, save=True):
        '''Compute and plot the field coordinates (P vs. Q) for the entire simulation

        Arguments:
            save (Optional[bool]): If true, save the figure. Defaults to True

        '''

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
        '''Compute and plot the field amplitudes (Ps,Qs) for the entire simulation

        Arguments:
            save (Optional[bool]): If true, save the figure. Defaults to True

        '''

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
        if save:
            plt.savefig('oscillations_110.pdf', bbox_inches='tight')


    def plot_coordinates(self, numPlot, IDs=None,save=True):
        '''Plot the particle coordinates history for a desired number of particles or specific particle IDs

        Arguments:
            numPlot (int): number of particles for which coordinates will be plotted <= num_particles
            IDs (Optional[array-like]): list of particle IDs to plot <=num_particles
            save (Optional[bool]): If true, save the figure. Defaults to True
        '''

        if not numPlot <= self.beam.num_particles:
            print "Number of particles specified cannot exceed number of simulated particles"
            raise

        if not IDs is None:
            z_vals = np.asarray(self.z_history)[:, IDs]
            y_vals = np.asarray(self.y_history)[:, IDs]
            x_vals = np.asarray(self.x_history)[:, IDs]
        else:
            z_vals = np.asarray(self.z_history)[:, :numPlot]
            y_vals = np.asarray(self.y_history)[:, :numPlot]
            x_vals = np.asarray(self.x_history)[:, :numPlot]

        fig = plt.figure(figsize=(12, 8))

        ax = fig.gca()

        for index,xs in enumerate(x_vals):
            x_lab = 'x{}'.format(index)
            ax.plot(np.asarray(self.tau_history) / c, xs, label=x_lab)
        for index,ys in enumerate(y_vals):
            y_lab = 'x{}'.format(index)
            ax.plot(np.asarray(self.tau_history) / c, ys, label=y_lab)
        for index, zs in enumerate(z_vals):
            z_lab = 'x{}'.format(index)
            ax.plot(np.asarray(self.tau_history) / c, zs, label=z_lab)

        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('Particle motion')
        ax.legend(loc='best')

        if save:
            plt.savefig('particle_coordinates.pdf', bbox_inches='tight')
