"""

eigenmodes.py is a file which contains the formula for computing the spatial eigenmodes of the cavity.
It also contains simple test functions and outputs for examining the cavity modes

For our simple example, consider spatial eigenmodes TMmnp for a rectangular cavity.
Assume that the cavity has dimensions [a,b,d]. The eigenmodes of the cavity
(corresponding to both the field eigenmodes and the vector potential eigenmodes)
are given by:

A_z(x,y,z) = C*sin(m*pi*x/a)*sin(n*pi*y/b)*cos(p*pi*z/d)

with normalization factor C = 2 np.sqrt(a*b*d) for p =/= 0.

Note that this function will assume a pre-specified single mode for now!


Nathan Cook
08/31/2016

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c


# cavity dimensions in meters
a = 0.1 #x plane
b = 0.1 #y plane
d = 0.2 #z plane

# Mode numbers
m = 1  # x mode
n = 1  # y mode
p = 0  # z mode


# Define mode frequency
omega_l = lambda m, n, p: np.pi * c * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
#Define frequency for simple example
OMEGA = omega_l(m,n,p)


# Define mode spatial eigenfunctions

#Normalization constants first
C_base = 2 / np.sqrt((a * b * d))
C_x = lambda m, n, p: C_base
C_y = lambda m, n, p: C_base * (m / n) * (b / a)  #differs by a ratio of wavenumbers (kx/ky) such that del*E = 0.
C_z = lambda m, n, p: C_base

#Spatial eigenmodes
A_x = lambda m, n, p, x, y, z: C_x(m, n, p) * np.cos(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.sin(
    p * np.pi * z / d)
A_y = lambda m, n, p, x, y, z: C_y(m, n, p) * np.sin(m * np.pi * x / a) * np.cos(n * np.pi * y / b) * np.sin(
    p * np.pi * z / d)
A_z = lambda m, n, p, x, y, z: C_z(m, n, p) * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.cos(
    p * np.pi * z / d)


#For p = 0. ONLY!
#int_A_x = lambda m,n,p,x,y,z: 0
#int_A_y = lambda m,n,p,x,y,z: 0
#int_A_z = lambda m,n,p,x,y,z: C_z(m,n,p)*z*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)

#Derivative quantities
#ddx_int_A_z = lambda m,n,p,x,y,z: C_z(m,n,p)*z*(m*np.pi/a)*np.cos(m*np.pi*x/a)*np.sin(n*np.pi*y/b)
#ddy_int_A_z = lambda m,n,p,x,y,z: C_z(m,n,p)*z*(n*np.pi/b)*np.sin(m*np.pi*x/a)*np.cos(n*np.pi*y/b)
#ddz_int_A_z = lambda m,n,p,x,y,z: C_z(m,n,p)*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)



#def compute_integrals(mode_list,x,y,z):
#    '''Return


def compute_wavenumbers(modes):
    '''Returns vectors of wavenumbers given vectors of mode numbers for each dimension

    Arguments:
        modes (ndArray): 3xL array of mode numbers (m,n,p) for L modes

    Returns:
]       ks (ndArray): 3xL array of mode wavenumbers (kx, ky, kz) for L modes

    '''
    ks = modes
    ks[:,0] = modes[:,0]*np.pi/a
    ks[:,1] = modes[:,1]*np.pi/b
    ks[:,2] = modes[:,2]*np.pi/d

    return ks


def old_compute_wavenumbers(m,n,p):
    '''Returns vectors of wavenumbers given vectors of mode numbers for each dimension

    Arguments:
        m (ndArray): vector of mode numbers m (length L)
        n (ndArray): vector of mode numbers n (length L)
        p (ndArray): vector of mode numbers p (length L)


    Returns:
        kx (ndArray): vector of mode wavenumbers kx (length L)
        ky (ndArray): vector of mode wavenumbers ky (length L)
        kz (ndArray): vector of mode wavenumbers kz (length L)

    '''

    kx = m*np.pi/a
    ky = n*np.pi/b
    kz = p*np.pi/d

    return kx,ky,kz


def dx_int_A_z(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of x derivative of int_A_z for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        dx_int_A_z (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)

    cos_product = np.cos(kxx) * np.sin(kyy)

    m_cos = np.einsum('i,ij->ij', kx, cos_product)

    z_m_cos = np.einsum('j,ij->ij', z, m_cos)

    return np.einsum('i,ij->ij', C_z(m, n, p), z_m_cos)


def dy_int_A_z(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of y derivative of int_A_z for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        dy_int_A_z (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)

    cos_product = np.sin(kxx) * np.cos(kyy)

    m_cos = np.einsum('i,ij->ij', ky, cos_product)

    z_m_cos = np.einsum('j,ij->ij', z, m_cos)

    return np.einsum('i,ij->ij', C_z(m, n, p), z_m_cos)


def dz_int_A_z(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of z derivative of int_A_z for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        dz_int_A_z (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)

    cos_product = np.sin(kxx) * np.sin(kyy)

    return np.einsum('i,ij->ij', C_z(m, n, p), cos_product)


def deriv_int_Ax(m,n,p,x,y,z):
    '''Placeholder function for the p = 0 case, so this is just an LxN array of zeros.'''

    L = len(m)
    N = len(x)

    return np.zeros((L, N))

def deriv_int_Ay(m,n,p,x,y,z):
    '''Placeholder function for the p = 0 case, so this is just an LxN array of zeros.'''

    L = len(m)
    N = len(x)

    return np.zeros((L, N))


def calc_int_A_z(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of int_A_z for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        int_A_z (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)

    sin_product = np.sin(kxx) * np.sin(kyy)

    z_sin = np.einsum('j,ij->ij', z, sin_product)

    return np.einsum('i,ij->ij', C_z(m, n, p), z_sin)


def calc_A_x(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of A_x for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        A_x (ndArray): An LxN array of values
    '''
    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)
    kzz = np.einsum('i,j->ij', kz, z)

    product = np.cos(kxx) * np.sin(kyy) * np.sin(kzz)

    return np.einsum('i,ij->ij', C_z(m, n, p), product)


def calc_A_y(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of A_y for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        A_y (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)
    kzz = np.einsum('i,j->ij', kz, z)

    product = np.sin(kxx) * np.cos(kyy) * np.sin(kzz)

    return np.einsum('i,ij->ij', C_z(m, n, p), product)


def calc_A_z(kx, ky, kz, x, y, z):
    '''
    Returns an LxN array of A_x for L modes evaluated at N particle positions.

    Arguments:
        kx (ndArray): vector of wavenumbers kx (length L)
        ky (ndArray): vector of wavenumbers ky (length L)
        kz (ndArray): vector of wavenumbers kz (length L)
        x (ndArray): vector of particle coordinates x (length N)
        y (ndArray): vector of particle coordinates y (length N)
        z (ndArray): vector of particle coordinates z (length N)

    Returns:
        A_z (ndArray): An LxN array of values
    '''

    kxx = np.einsum('i,j->ij', kx, x)
    kyy = np.einsum('i,j->ij', ky, y)
    kzz = np.einsum('i,j->ij', kz, z)

    product = np.sin(kxx) * np.sin(kyy) * np.cos(kzz)

    return np.einsum('i,ij->ij', C_z(m, n, p), product)