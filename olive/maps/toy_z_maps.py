# Maps module for OLIVE
#
# Maps consist of functions that operate on beam and field class instances.
#
#
# Units
#   -Assume CGS units for now
#

#Base map class can have a get_beam function
#This will be a set of functions rather than a class structure to allow for more streamlining of operations!
#beam and fields will provide the necessary inputs?

import numpy as np
from scipy.constants import c


class Map(object):

    def __init__(self):
        '''Empty initializer for Map object'''


    def update_q(self, beam, h, k=0, step=1.):
        '''Update for all particles a single component qk -> qk+1(or 1/2) given the momentum pk

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            h (Float): step size in units cm
            k (int): Index of coordinate being advance (k=0 for x, k=1 for y, k=2 for z)
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''
        if k == 0:
            beam.x = beam.x + step * h * beam.px / beam.gmc
        elif k == 1:
            beam.y = beam.y + step * h * beam.py / beam.gmc
        elif k == 2:
            beam.z = beam.z + step * h * beam.pz / beam.gmc
        else:
            raise ValueError("Coordinate index outside of range [0,1,2]")


    def kick_p(self, beam, fields, h, k=0, sign=1, step=1.):
        '''Kick p is the kick portion of the coupling Hamiltonian map, which updates each component of p as well
         as the field momentum P. The kick remains dependent upon the coordinate subscript k, as it differs for each
         coordinate-specific map x,y,z.

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            k (int): Index of coordinate dictating the advance (k=0 for x, k=1 for y, k=2 for z)
            sign (int): 1 if subtracting (e.g. first step), -1 if adding (e.g. 2nd step)
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        if k == 0:
            # k = 0 means evaluating A_x for all field couplings
            # Each function returns an LxN array - integral values for L modes evaluated at N particle positions.
            ddx_int_A_x = np.zeros((fields.num_modes, beam.num_particles))
            ddy_int_A_x = np.zeros((fields.num_modes, beam.num_particles))
            ddz_int_A_x = np.zeros((fields.num_modes, beam.num_particles))

            # LxN array->int_A_x evaluate for L modes at N particle positions
            int_A_x = np.zeros((fields.num_modes, beam.num_particles))

            beam.px = beam.px - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddx_int_A_x)
            beam.py = beam.py - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddy_int_A_x)
            beam.pz = beam.pz - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddz_int_A_x)

            # Update modementa
            # array of L x N (L modes and N particles)
            # sum over each particle for all l modes -> array of length l
            fields.P = fields.P - sign * step * h * (1 / c) * np.einsum('ij->i', beam.qs * int_A_x)

        elif k == 1:
            # k = 1 means evaluating A_y for all field couplings
            ddx_int_A_y = np.zeros((fields.num_modes, beam.num_particles))
            ddy_int_A_y = np.zeros((fields.num_modes, beam.num_particles))
            ddz_int_A_y = np.zeros((fields.num_modes, beam.num_particles))

            # LxN array->int_A_y evaluate for L modes at N particle positions
            int_A_y = np.zeros((fields.num_modes, beam.num_particles))

            beam.px = beam.px - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddx_int_A_y)  # fields.Q*ddx_int_A_y
            beam.py = beam.py - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddy_int_A_y)  # fields.Q*ddy_int_A_y
            beam.pz = beam.pz - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddz_int_A_y)  # fields.Q*ddz_int_A_y

            # Update modementa
            # array of L x N (L modes and N particles)
            # sum over each particle for all l modes -> array of length l
            fields.P = fields.P - sign * step * h * (1 / c) * np.einsum('ij->i', beam.qs * int_A_y)


        elif k == 2:
            # k = 2 means evaluating A_z for all field couplings
            # Returns an LxN array - integral values for L modes evaluated at N particle positions.
            ddx_int_A_z = fields.dx_int_A_z(beam.x, beam.y, beam.z)
            ddy_int_A_z = fields.dy_int_A_z(beam.x, beam.y, beam.z)
            ddz_int_A_z = fields.dz_int_A_z(beam.x, beam.y, beam.z)

            # LxN array->int_A_z evaluate for L modes at N particle positions
            int_A_z = fields.calc_int_A_z(beam.x, beam.y, beam.z)

            beam.px = beam.px - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddx_int_A_z)  # fields.Q*ddx_int_A_z
            beam.py = beam.py - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddy_int_A_z)  # fields.Q*ddy_int_A_z
            beam.pz = beam.pz - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddz_int_A_z)  # fields.Q*ddz_int_A_z

            # Update modementa
            # array of L x N (L modes and N particles)
            # sum over each particle for all l modes -> array of length l
            # move q term into int term
            fields.P = fields.P - sign * step * h * (1 / c) * np.einsum('ij->i', beam.qs * int_A_z)

        else:
            raise ValueError("Coordinate index outside of range [0,1,2]")


    def update_x(self, beam, fields, h, step=1.):
        '''Perform the map for x, consisting of a half kick, drift, then half kick

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        self.kick_p(beam, fields, h, k=0, sign=1, step=0.5 * step)
        self.update_q(beam, h, k=0, step=step)
        self.kick_p(beam, fields, h, k=0, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


    def update_y(self, beam, fields, h, step=1.):
        '''Perform the map for y consisting of a half kick, drift, then half kick

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        self.kick_p(beam, fields, h, k=1, sign=1, step=0.5 * step)
        self.update_q(beam, h, k=1, step=step)
        self.kick_p(beam, fields, h, k=1, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


    def update_z(self, beam, fields, h, step=1.):
        '''Perform the map for y consisting of a half kick, drift, then half kick

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        self.kick_p(beam, fields, h, k=2, sign=1, step=0.5 * step)
        self.update_q(beam, h, k=2, step=step)
        self.kick_p(beam, fields, h, k=2, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


    def rotate_fields_determinant(self, fields, h, step=1.):
        '''Update field phases self consistently with the time step.

        Arguments:
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        Note that this step applies a fixed rotation operator that only varies on the size
        of the time step.

        '''
        #define placeholders
        currentQ = fields.Q
        currentP = fields.P

        rot_mat = np.asarray([[np.cos(fields.omegas * 1 * h),
                               -1. * fields.omegas * fields.M * np.sin(fields.omegas * 1. * h)],
                              [(1 / (fields.M * fields.omegas)) * np.sin(fields.omegas * 1. * h),
                               np.cos(fields.omegas * 1. * h)]])
        #print rot_mat.shape

        # append the determinant of the rotation matrix
        fields.det.append(np.asarray([np.linalg.slogdet(rot_mat[:,:,i]) for i in range(rot_mat.shape[-1])]))

        #update Q then P using placeholders
        fields.Q = currentQ * np.cos(fields.omegas * step * h) + currentP * (1 / (fields.M * fields.omegas)) * np.sin(
            fields.omegas * step * h)

        fields.P = currentP * np.cos(fields.omegas * step * h) - fields.omegas * fields.M * currentQ * np.sin(
            fields.omegas * step * h)


    def rotate_fields(self, fields, h, step=1.):
        '''Update field phases self consistently with the time step.

        Arguments:
            fields (olive.fields.field.Field): an Olive object containing field information
            h (Float): step size in units cm
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        Note that this step applies a fixed rotation operator that only varies on the size
        of the time step.

        '''
        #define placeholders
        currentQ = fields.Q
        currentP = fields.P


        #update Q then P using placeholders
        fields.Q = currentQ * np.cos(fields.omegas * step * h) + currentP * (1 / (fields.M * fields.omegas)) * np.sin(
            fields.omegas * step * h)

        fields.P = currentP * np.cos(fields.omegas * step * h) - fields.omegas * fields.M * currentQ * np.sin(
            fields.omegas * step * h)


    def initialize_beam(self, beam, fields):
        '''Converts beam momenta to canonical and calculates needed gamma_m_c quantities

        Arguments:
            beam (olive.particles.beam.Beam): an Olive object containing beam information
            fields (olive.fields.field.Field): an Olive object containing field information

        '''

        beam.convert_mechanical_to_canonical(fields)
        beam.calc_gamma_m_c(fields)