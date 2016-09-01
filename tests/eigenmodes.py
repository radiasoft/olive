"""

eigenmodes.py is a file which contains the formula for computing the spatial eigenmodes of the cavity.
It also contains simple test functions and outputs for examining the cavity modes

For our simple example, consider spatial eigenmodes TMmnp for a rectangular cavity.
Assume that the cavity has dimensions [a,b,d]. The eigenmodes of the cavity
(corresponding to both the field eigenmodes and the vector potential eigenmodes)
are given by:

f_z(x,y,z) = C*sin(m*pi*x/a)*sin(n*pi*y/b)*cos(p*pi*z/d)

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

# Define mode spatial eigenfunction

#C-value depends upon
if p == 0:
    C_mode = 2 / np.sqrt((a * b * d))
else:
    C_mode = 2 * np.sqrt(2) / np.sqrt((a * b * d))

#full eigenmode component in z
f_z = lambda m, n, p, x, y, z: C_mode * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.cos(
    p * np.pi * z / d)

#110 fundamental mode
f_z_110 = lambda x,y,z: C_mode * np.sin(np.pi * x / a) * np.sin(np.pi * y / b)
f_z_110_dx = lambda x,y,z: C_mode * (np.pi/a) * np.cos(np.pi * x / a) * np.sin(np.pi * y / b)
f_z_110_dy = lambda x,y,z: C_mode * (np.pi/b) * np.sin(np.pi * x / a) * np.cos(np.pi * y / b)
f_z_110_dz = lambda x,y,z: 0. #constant accelerating field along z means constant vector potential along z


def f_mode(x,y,z):
    '''Return an array of spatial eigenmode vectors at position (x,y,z)
        Arguments:
            x,y,z (ndarray): arrays of particle positions

        Returns:
            fx,fy,fz (ndarray): arrays of eigenmode evaluations and
                                spatial derivatives - each length

    '''

    ####------------------INSERT FUNCTION HERE-------------------####


    fz = f_z_110(x,y,z)
    fz_dx = f_z_110_dx(x,y,z)
    fz_dy = f_z_110_dy(x,y,z)
    fz_dz = f_z_110_dz(x,y,z)


    return fz, fz_dx, fz_dy, fz_dz