"""

simple_run is an example file which demonstrates the OLIVE algorithm using 4 particles and 3 modes
in a rectangular cavity. The example demonstrates the basic sequencing and usage of hte OLIVE codebase
as well as a few simple diagnostics.

Nathan Cook
Last Updated: 7/24/2017
Initial: 09/08/2016

Sequencing:
-----------

Map approach: A single step is the concatenation of the following operators:

M_R(1/2) M_x(1/2) M_y(1/2) M_z(1/2) M_y(1/2) M_x(1/2) M_R(1/2)

where M_R expresses the action of the field Hamiltonian, which is in effect a harmonic oscillator. This
can be represented by a simple rotation, as before.

The different M_x type maps represent a sequence of kicks (momenta updates) and drifts (coordinate updates) wherein
only the coordinate specified by the subscript (x,y,z) is updated. An example is as follows:


M_x = Kick_ps(all of them 4-updates), Drift_x, Kick_ps(all of them again).

Note that the 1st kickP and the second kickP differ in FORM by a minus sign due to the similarity transform done to
produce the kicks from the Hamiltonian. Also note that even though the form of the computations are the same, they
must be re-evaluated using the new x-coordinate that was updated by the drift operation.


Initialization requirements:
- We are given mechanical momentum, and all relevant field quantities.
- Field quantities include eigenmodes of interest (for pillbox this means specifying mode #s l = (m,n,p) and initial
strengths Q_l.
- We must compute frequencies as desired, and produce an array of mode information corresponding to a
mode index (l) as we see fit. (e.g. l = 0 -> TM mode with m,n,p = 1,1,0 and has corresponding initial Q_l and P_l).

Beginning the algorithm requires transforming the mechanical momentum p to canonical momentum P = p + (e/c)A.


Usage:
------
python simple_run.py
"""

import numpy as np
import math
import time
import itertools
import matplotlib as mpl
mpl.use('TkAgg') #for compliance with virtual environments on OSX
import matplotlib.pyplot as plt
from scipy.constants import e, k
from scipy.constants import c as c_mks
from scipy.constants import m_e as me_mks
from scipy.constants import m_p as mp

from olive.fields import eigenmodes
from olive.particles import beam
from olive.fields import field
from olive.maps import toy_z_maps
from olive.interface import simulator

# Set the default mass and charge for an electron
m = me_mks*1.e3 #cgs
q = 4.80320451e-10 #esu 1.*e
c = c_mks*1.e2 #cgs

q_over_c = q/c


### Define the cavity

a = 10. #cavity dimensions in cm
b = 10.
d = 20.

cavity = eigenmodes.RectangularModes(a,b,d)
myFields = field.Field(cavity)

initialA0 = 3668.25645
Q0 = initialA0 * np.asarray([1., 0., 0.])  # normalize the amplitude here
P0 = np.asarray([0., 0., 0.])
mode_nums = np.asarray([[1, 1, 0], [2, 1, 0],[1, 2, 0]]) #, [3,1,0],[4,1,0]])

myFields.create_modes(mode_nums,Q0,P0)


### Initialize the beam particles

myBunch = beam.Beam()


########---------Specify initial conditions for particles--------------######

gamma = 50.  # gamma =50 electrons
beta = np.sqrt(1. - 1. / gamma ** 2)
# starting coordinates
q1 = [a / 2., b / 2., 0.]
q2 = [a / 4., b / 2., 0.]
q3 = [a / 2., b / 4., 0.]
q4 = [a / 4., b / 4., 0.]

q0 = np.asarray([q1, q2, q3, q4])

# define weights
wf = 10000. #3.0e11 #10000.
weights = wf * np.ones(len(q0))

# Our canonical momentum is now an actual momentum
p_vec = [0., 0., 1.]  # all z momentum
p1 = gamma * beta * m * c * np.asarray(p_vec)

p0 = np.asarray([p1, p1, p1, p1])

myBunch.add_bunch(q0,p0,weights)


### Initialize the simulator and run

myMaps = toy_z_maps.Map()
mySimulator = simulator.simulator(myBunch,myFields,myMaps)

maxTau = d/beta #traverse one cavity length
numSteps = 5e2

mySimulator.define_simulation(maxTau,numSteps)

mySimulator.step(numSteps)


### Diagnostics

#Diagnostics every 10 steps
dEpercent = mySimulator.energy_change_sytem()
dEtotal = mySimulator.energy_change_sytem_ergs()
print "Energy change for {:e} timesteps is {}%".format(numSteps, dEpercent)



save=True

mySimulator.plot_total_system_energy(save)
mySimulator.plot_mode_energies(save)
mySimulator.plot_field_rotations(save)

