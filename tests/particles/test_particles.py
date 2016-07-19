import pytest
import numpy as np
from olive.particles import particle
from scipy.constants import m_e, e, c


#dictionary for passing to values
_PD = {
    'num_p' : 100,
    'mass' : m_e,
    'dim' : 3,
    'z_range' : 1,
    'pz_range' : 1e-3,
    'expected_gamma' : np.sqrt(2),
    'expected_beta' : 1./np.sqrt(2)
}

_EPSILON = 1e-15 #minimum threshold for double precision

def _assert(expect, actual):

    assert abs(expect-actual) <= _EPSILON, \
    'expected value {} != {} actual value'.format(expect, actual)

    #if abs(expect-actual) > _EPSILON:
    #    raise AssertionError(
    #        'expected value {} != {} actual value'.format(expect, actual))
    #else:
    #    return

def _assert_array(expect, actual):
    if np.shape(expect):
        for e, a in zip(expect, actual):
            _assert(e, a)
    else:
        _assert(expect, actual)


def symmetric_array(num, m):
    """
    Warps np.linspace to return a symmetric array ranging from -m : m with num entries
    """

    return np.linspace(-1.*m,1.*m,num)

#@pytest.fixture
def bunch():
    tb = particle.Particle(_PD)

    #particles have zero off-axis amplitude
    x = np.zeros(_PD['num_p'])
    y = x.copy()
    z = symmetric_array(_PD['num_p'],_PD['z_range'])
    positions = np.asarray(zip(x,y,z))

    #particles have zero transverse momenta
    px = np.zeros(_PD['num_p'])
    py = px.copy()
    #slower particles are placed ahead of faster particles
    pz = (1.-symmetric_array(_PD['num_p'],_PD['z_range']))*m_e*c
    momenta = np.asarray(zip(px,py,pz))

    tb.add_bunch(positions,momenta)

    return tb

#mybunch = bunch()

#REPLACE CLASS HERE -> Just do a list of functions
#class TestBunch(object):

def test_gamma():
    mybunch = bunch()
    _assert(_PD['expected_gamma'], mybunch.compute_gamma_z())

def test_beta():
    mybunch = bunch()
    _assert(_PD['expected_beta'], mybunch.compute_beta_z())
