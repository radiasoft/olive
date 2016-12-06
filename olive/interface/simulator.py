# -*- coding: utf-8 -*-
u"""Interface class for creating and running simulations using OLIVE
:copyright: Copyright (c) 2016 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""

from olive.fields import field_data


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
        self.tau_history = []

        self.beam.convert_mechanical_to_canonical(self.fields)
        self.gmc_history = []

        # Construct history arrays to store updates
        self.x_history = [self.beam.x]
        self.y_history = [self.beam.y]
        self.z_history = [self.beam.z]
        self.px_history = [self.beam.px]
        self.py_history = [self.beam.py]
        self.pz_history = [self.beam.pz]


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
            # Initial half step field rotation
            self.beam.calc_gamma_m_c(self.fields)
            self.maps.rotate_fields(self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_z(self.beam, self.fields, h, step=1.0)
            self.maps.update_y(self.beam, self.fields, h, step=0.5)
            self.maps.update_x(self.beam, self.fields, h, step=0.5)
            self.maps.rotate_fields(self.fields, step=0.5)

            # update times for diagnostics
            self.tau = self.tau + h
            self.tau_history.append(self.tau)

            # update coordinate histories
            self.update_histories()

            i = i - 1
