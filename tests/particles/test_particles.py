
from olive.particles import ptcl
import pytest
from scipy.constants import c,e,m_e
import numpy as np

#construct dictionary for initialization
pd = {}
pd['num_particles'] = 100
pd['mass'] = m_e
pd['dim'] = 3


positions = np.linspace(-1.,1.,pd['num_p'])
momenta = (1+np.linsapce(-1e-3,1e-3,pd['num_p']))*m_e*c

def test_bunch():
    tb = ptcl.ptcl(pd)

    tb.add_bunch(positions,momenta)

    tb.compute_beta_z()

    #assert ptcl.mass = m_e
