
from olive.particles import ptcl
import pytest
from scipy.constants import c,e,m_e
import numpy as np

#dictionary for passing to values
pd = {}
pd['num_particles'] = 100
pd['mass'] = m_e
pd['dim'] = 3
pd['z_range'] = 1
pd['pz_range'] = 1e-3
pd['expected_gamma'] = np.sqrt(2)
pd['expected_beta'] = 1./np.sqrt(2)

_EPSILON = 1e-15 #minimum threshold for double precision

def _assert(expect, actual):
    if abs(expect-actual) > _EPSILON:
        raise AssertionError(
            'expected value {} != {} actual value'.format(expect, actual))
    else:
        return

def _assert_array(expect, actual):
    if np.shape(expect):
        for e, a in zip(expect, actual):
            _assert(e, a)
    else:
        _assert(expect, actual)


@pytest.fixture
def bunch():
    tb = ptcl.Ptcl(pd)

    #particles have zero off-axis amplitude
    x = np.zeros(pd['num_p'])
    y = np.zeros(pd['num_p'])
    z = np.linspace(-1,1,pd['num_p'])
    positions = np.asarray(zip(x,y,z))

    #particles have zero transverse momenta
    px = np.zeros(pd['num_p'])
    py = np.zeros(pd['num_p'])
    pz = (1.-np.linspace(-1e-3,1e3,pd['num_p']))*m_e*c #slower particles are placed ahead of faster particles
    momenta = np.asarray(zip(px,py,pz))

    tb.add_bunch(positions,momenta)

    return tb


class TestBunch:

    def test_gamma(self,bunch):
        _assert(pd['expected_gamma'], bunch.compute_gamma_z())

    def test_beta(self, bunch):
        _assert(pd['expected_beta'], bunch.compute_beta_z())
