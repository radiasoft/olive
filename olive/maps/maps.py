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


def update_q(beam,h,k=0, step=1.):
    '''Update for all particles a single component qk -> qk+1(or 1/2) given the momentum pk

    Arguments:
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


def kick_p(beam,fields,h, k=0, sign=1, step=1.):
    '''Kick p is the kick portion of the coupling Hamiltonian map, which updates each component of p as well
     as the field momentum P. The kick remains dependent upon the coordinate subscript k, as it differs for each
     coordinate-specific map x,y,z.

    Arguments:
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

        # print ddx_int_A_x.shape

        # LxN array->int_A_x evaluate for L modes at N particle positions
        int_A_x = np.zeros((fields.num_modes, beam.num_particles))

        # np.dot(fields.Q,ddy_int_A_x) is the same as np.einsum('i,ij->j',Q,dxintA_L)

        beam.px = beam.px - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddx_int_A_x)
        beam.py = beam.py - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddy_int_A_x)
        beam.pz = beam.pz - sign * step * h * (beam.qs / c) * np.dot(fields.Q, ddz_int_A_x)

        # Update modementa
        # sum over each particle for all l modes -> array of length l
        fields.P = fields.P - sign * step * h * (1 / c) * np.einsum('ij->i', beam.qs * int_A_x)

    elif k == 1:
        # k = 1 means evaluating A_y for all field couplings
        ddx_int_A_y = np.zeros((fields.num_modes, beam.num_particles))
        ddy_int_A_y = np.zeros((fields.num_modes, beam.num_particles))
        ddz_int_A_y = np.zeros((fields.num_modes, beam.num_particles))

        # LxN array->int_A_y evaluate for L modes at N particle positions
        int_A_y = np.zeros((fields.num_modes, beam.num_particles))

        beam.px = beam.px - sign * step * self.h * (beam.qs / c) * np.dot(fields.Q, ddx_int_A_y)  # fields.Q*ddx_int_A_y
        beam.py = beam.py - sign * step * self.h * (beam.qs / c) * np.dot(fields.Q, ddy_int_A_y)  # fields.Q*ddy_int_A_y
        beam.pz = beam.pz - sign * step * self.h * (beam.qs / c) * np.dot(fields.Q, ddz_int_A_y)  # fields.Q*ddz_int_A_y

        # Update modementa
        # array of L x N (L modes and N particles)
        # sum over each particle for all l modes -> array of length l
        fields.P = fields.P - sign * step * self.h * (1 / c) * np.einsum('ij->i', beam.qs * int_A_y)


    elif k == 2:
        # k = 2 means evaluating A_z for all field couplings
        # Returns an LxN array - integral values for L modes evaluated at N particle positions.
        ddx_int_A_z = dx_int_A_z(fields.modes[:, 0], fields.modes[:, 1], fields.modes[:, 2], beam.x, beam.y, beam.z)
        ddy_int_A_z = dy_int_A_z(fields.modes[:, 0], fields.modes[:, 1], fields.modes[:, 2], beam.x, beam.y, beam.z)
        ddz_int_A_z = dz_int_A_z(fields.modes[:, 0], fields.modes[:, 1], fields.modes[:, 2], beam.x, beam.y, beam.z)

        # LxN array->int_A_z evaluate for L modes at N particle positions
        int_A_z = calc_int_A_z(fields.modes[:, 0], fields.modes[:, 1], fields.modes[:, 2], beam.x, beam.y, beam.z)

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


def update_x(beam,fields, h, step=1.):
    '''Perform the map for x, consisting of a half kick, drift, then half kick'''

    kick_p(beam,fields,h,k=0, sign=1, step=0.5 * step)
    update_q(beam,fields,h,k=0, step=step)
    kick_p(beam,fields,h,k=0, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


def update_y(beam,fields, h, step=1.):
    '''Perform the map for y consisting of a half kick, drift, then half kick'''

    kick_p(beam,fields,h,k=1, sign=1, step=0.5 * step)
    update_q(beam,fields,h,k=1, step=step)
    kick_p(beam,fields,h,k=1, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


def update_z(beam,fields, h, step=1.):
    '''Perform the map for y consisting of a half kick, drift, then half kick'''

    kick_p(beam,fields, h,k=2, sign=1, step=0.5 * step)
    update_q(beam,fields, h,k=2, step=step)
    kick_p(beam,fields, h,k=2, sign=-1, step=0.5 * step)  # reverse the sign for the 2nd kick due to similarity transform


def rotate_fields(step=1.):
    '''Update field phases self consistently with the time step.

    Arguments:
        step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

    Note that this step applies a fixed rotation operator that only varies on the size
    of the time step.

    '''
    currentQ = fields.Q
    currentP = fields.P

    print currentQ

    fields.Q = currentQ * np.cos(fields.omegas * step * h) + currentP * (1 / (fields.M * fields.omegas)) * np.sin(
        fields.omegas * step * h)

    print currentQ

    fields.P = currentP * np.cos(fields.omegas * step * h) - fields.omegas * fields.M * currentQ * np.sin(
        fields.omegas * step * h)
