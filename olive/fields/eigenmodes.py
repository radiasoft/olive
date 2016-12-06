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

# cavity dimensions in meters
#a = 50. #*1e-2 #x plane
#b = 40. #1e-2 #y plane
#d = 10. #1e-2 #z plane

# Mode numbers
m = 1  # x mode
n = 1  # y mode
p = 0  # z mode



import numpy as np
from scipy.constants import c as c_mks

c = c_mks*1.e2


class RectangularModes(object):
    '''Class containing rectangular mode constructors and relevant functions

    In the future - this can be made into a subclass for use with cylindrical cavities, and ultimately
    generalized for polynomial eigenmode expansions.

    '''

    def __init__(self, a,b,d):

        '''Constructor for modes - needs only dimensions

        Arguments:
            a (float): cavity length in x-plane
            b (float): cavity length in y-plane
            d (float): cavity length in z-plane (presumed longitudinal direction)

        '''

        self.x = a
        self.y = b
        self.z = d

        self.M = 1. / (16. * np.pi * c) * (a * b * d) #M-factor for integration of field quantities

        C_base = 1.
        self.C_x = lambda m, n, p: C_base * m / m
        self.C_y = lambda m, n, p: C_base * (m / m)  # differs by a ratio of wavenumbers (kx/ky) such that del*E = 0.
        self.C_z = lambda m, n, p: C_base * m / m


    def get_mode_frequencies(self,m,n,p):
        '''Return mode (angular) frequencies for the cavity

        Arguments:
            m (int): x-plane eigenvalue
            n (int): y-plane eigenvalue
            p (int): z-plane eigenvalue

        '''

        return np.pi * c * np.sqrt((m/self.x) ** 2 + (n/self.y) ** 2 + (p/self.z) ** 2)

    def get_mode_wavenumbers(self,m,n,p):

        '''Return mode wavenumbers for the modes

        Arguments:
            m (int): x-plane eigenvalue
            n (int): y-plane eigenvalue
            p (int): z-plane eigenvalue

        '''

        #ks = np.zeros()
        kxs = 1.0 * m * np.pi / self.x
        kys = 1.0 * n * np.pi / self.y
        kzs = 1.0 * p * np.pi / self.z

        return np.asarray([kxs,kys,kzs])

    def get_mode_Ms(self,m,n,p):

        '''Return mode Ml quantities for the modes

        Arguments:
            m (int): x-plane eigenvalue
            n (int): y-plane eigenvalue
            p (int): z-plane eigenvalue

        '''

        return self.M*np.ones(len(m))

    def get_mode_Ks(self,m,n,p):

        '''Return mode Ml quantities for the modes

        Arguments:
            m (int): x-plane eigenvalue
            n (int): y-plane eigenvalue
            p (int): z-plane eigenvalue

        '''

        return self.Ml*(m**2 + n**2)


    def calc_A_x(self, kx, ky, kz, x, y, z):
        '''
        Returns an LxN array of A_x for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
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

        return np.einsum('i,ij->ij', self.C_x(kx, ky, kz), product)

    def calc_A_y(self, kx, ky, kz, x, y, z):
        '''
        Returns an LxN array of A_y for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
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

        return np.einsum('i,ij->ij',self.C_y(kx, ky, kz), product)

    def calc_A_z(self, kx, ky, kz, x, y, z):
        '''
        Returns an LxN array of A_x for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
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

        return np.einsum('i,ij->ij', self.C_z(kx, ky, kz), product)


    def dx_int_A_z(self,ks, x, y, z):
        '''
        Returns an LxN array of x derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dx_int_A_z (ndArray): An LxN array of values
        '''
        kx = ks[0]
        ky = ks[1]
        kz = ks[2]

        kxx = np.einsum('i,j->ij', kx, x)
        kyy = np.einsum('i,j->ij', ky, y)

        cos_product = np.cos(kxx) * np.sin(kyy)

        m_cos = np.einsum('i,ij->ij', kx, cos_product)

        z_m_cos = np.einsum('j,ij->ij', z, m_cos)

        return np.einsum('i,ij->ij', self.C_z(kx, ky, kz), z_m_cos)

    def dy_int_A_z(self,ks, x, y, z):
        '''
        Returns an LxN array of y derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dy_int_A_z (ndArray): An LxN array of values
        '''
        kx = ks[0]
        ky = ks[1]
        kz = ks[2]


        kxx = np.einsum('i,j->ij', kx, x)
        kyy = np.einsum('i,j->ij', ky, y)

        cos_product = np.sin(kxx) * np.cos(kyy)

        m_cos = np.einsum('i,ij->ij', ky, cos_product)

        z_m_cos = np.einsum('j,ij->ij', z, m_cos)

        return np.einsum('i,ij->ij', self.C_z(kx, ky, kz), z_m_cos)

    def dz_int_A_z(self,ks, x, y, z):
        '''
        Returns an LxN array of z derivative of int_A_z for L modes evaluated at N particle positions.

        Arguments:
            ks (ndArray): Lx3 array of wavenumbers
            x (ndArray): vector of particle coordinates x (length N)
            y (ndArray): vector of particle coordinates y (length N)
            z (ndArray): vector of particle coordinates z (length N)

        Returns:
            dz_int_A_z (ndArray): An LxN array of values
        '''

        kx = ks[0]
        ky = ks[1]
        kz = ks[2]

        kxx = np.einsum('i,j->ij', kx, x)
        kyy = np.einsum('i,j->ij', ky, y)

        cos_product = np.sin(kxx) * np.sin(kyy)

        return np.einsum('i,ij->ij', self.C_z(kx, ky, kz), cos_product)

    def calc_int_A_z(self, ks, x, y, z):
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
        kx = ks[0]
        ky = ks[1]
        kz = ks[2]

        kxx = np.einsum('i,j->ij', kx, x)
        kyy = np.einsum('i,j->ij', ky, y)

        sin_product = np.sin(kxx) * np.sin(kyy)

        z_sin = np.einsum('j,ij->ij', z, sin_product)

        return np.einsum('i,ij->ij', self.C_z(kx, ky, kz), z_sin)
