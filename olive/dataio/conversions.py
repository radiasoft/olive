import numpy as np
from scipy.constants import m_e as m
from scipy.constants import c, e
from scipy.constants import physical_constants


m_e = physical_constants['electron mass energy equivalent in MeV'][0]


def convert_units_olive2elegant(x, px, y, py, z, pz):
    '''
    Convert bunch phase space coordinates from Olive units (CGS) to Elegant units

    Arguments:
        x (float): x coordinate - cm
        px (float): x momentum coordinate - g cm /s
        y (float): y coordinate - cm
        py (float): py momentum coordinate - g cm /s
        z (float): x coordinate - cm
        pz (float): x momentum coordinate - g cm /s

    '''


    bunch = np.column_stack([x, px, y, py, z, pz])
    #print bunch.shape
    new_bunch = np.empty_like(bunch)

    total_momentum = np.sqrt(px**2 + py**2 + pz**2)
    new_bunch[:, [1, 3]] = bunch[:, [1, 3]] / bunch[:, [5]]  # Convert transverse momenta to angles
    new_bunch[:, 5] = total_momentum * c / e * 1e-11  # eV /c = 1 g cm / s
    new_bunch[:, 5] = new_bunch[:, 5] / m_e  # Convert to elegant's m_e*c for momentum

    betas = new_bunch[:, 5] / np.sqrt(1. + new_bunch[:, 5]**2)
    new_bunch[:, [0, 2]] = bunch[:, [0, 2]] / 100.  # Convert cm to m
    new_bunch[:, 4] = -bunch[:, 4] / (betas * c) / 100.  # Flip head/tail and convert to t = s/(beta * c)

    return new_bunch


def convert_units_elegant2olive(bunch, tcenter=None):
    '''
    Convert bunch phase space coordinates from Elegant units to Olive units (CGS)

    Arguments:
        x (float): x coordinate - m
        px (float): x momentum coordinate - MeV/c
        y (float): y coordinate - m
        py (float): py momentum coordinate - MeV/c
        z (float): x coordinate - m
        pz (float): x momentum coordinate - MeV/c

    '''

    new_bunch = np.empty_like(bunch)

    new_bunch[:, 5] = bunch[:, 5] / np.sqrt(bunch[:, 1]**2 + bunch[:, 3]**2 + 1.)  # Ptot to pz
    new_bunch[:, [1, 3]] = bunch[:, [1, 3]] * new_bunch[:, [5]]  # xp,yp to px,py
    new_bunch[:, [1, 3, 5]] = new_bunch[:, [1, 3, 5]] * m_e * e / c * 1e11  # Convert to MeV/c then to g cm / s

    if tcenter:
        bunch[:, 4] = -bunch[:, 4] - tcenter  # Send reverse direction and 0 on CoM
    else:
        bunch[:, 4] = -bunch[:, 4] - (np.average(-bunch[:, 4]))

    betas = bunch[:, 5] / np.sqrt(1. + bunch[:, 5] ** 2)

    new_bunch[:, [0, 2]] = bunch[:, [0, 2]] * np.array([100., 100.])  # Convert to cm
    new_bunch[:, 4] = bunch[:, 4] * 100. * c * betas  # coordinate s = beta * c * t

    return new_bunch
