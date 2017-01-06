#Test a single step of an OLIVE simulation for varying step size

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import e, k
from scipy.constants import c as c_mks
from scipy.constants import m_e as me_mks
from scipy.constants import m_p as mp

# Set the default mass and charge for an electron
m = me_mks*1.e3 #cgs
q = 4.80320451e-10 #esu 1.*e
c = c_mks*1.e2 #cgs

q_over_c = q/c

#OLIVE imports
from olive.fields import eigenmodes
from olive.particles import beam
from olive.fields import field
from olive.maps import toy_z_maps
from olive.interface import simulator

def one_olive_step(numSteps):
    '''
    Construct a basic OLIVE simulation and compute energy deviation with step size
    
    Arguments:
        num_Steps (int): number of steps to perform (for fixed cavity length) - gets coupled to a stepSize
        
    Returns:
        ds (float): step size for single step performed
        deTotal (float): total system energy deviation
    
    '''
    #cavity dimensions in cm
    a = 10.
    b = 10.
    d = 20.

    cavity = eigenmodes.RectangularModes(a,b,d)
    
    myBunch = beam.Beam()
    myFields = field.Field(cavity)

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
    wf = 1.e7 #10000. #3.0e11 #10000.
    weights = wf * np.ones(len(q0))

    # Our canonical momentum is now an actual momentum
    p_vec = [0., 0., 1.]  # all z momentum
    p1 = gamma * beta * m * c * np.asarray(p_vec)

    p0 = np.asarray([p1, p1, p1, p1])

    myBunch.add_bunch(q0,p0,weights)

    initialA0 = 3668.25645
    Q0 = initialA0 * np.asarray([1., 0., 0.])  # normalize the amplitude here
    P0 = np.asarray([0., 0., 0.])
    mode_nums = np.asarray([[1, 1, 0], [2, 1, 0],[1, 2, 0]]) #, [3,1,0],[4,1,0]])

    myFields.create_modes(mode_nums,Q0,P0)

    myMaps = toy_z_maps.Map()
    mySimulator = simulator.simulator(myBunch,myFields,myMaps)

    maxTau = d/beta #traverse one cavity length
    ds = maxTau/numSteps
    mySimulator.define_simulation(maxTau,numSteps)

    mySimulator.step(1)
    
    dEtotal = mySimulator.energy_change_sytem()
    print "Energy change for {} timesteps is {}".format(numSteps,dEtotal)
    
    return ds, dEtotal
    
    
    
#Run the script a few times to generate some data points

stepVals = [10,100,500,1000,5000,10000,20000,50000,75000,100000,1.5e5,2e5,5e5]
dEvals = []
dsVals = []

for nums in stepVals:
    
    ds,dE = one_olive_step(nums)
    dsVals.append(ds)
    dEvals.append(dE)
    
ds = np.asarray(dsVals)
Es = np.asarray(dEvals)


#Produce plot - uncomment bottom line to save
with mpl.style.context('rs_paper'):
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(1./ds, 5.6e-4*ds*ds, label=r'~$1/dt^2$')
    ax.loglog(1./ds, 5.6e-4*ds*ds*ds, label=r'~$1/dt^3$')
    ax.loglog(1./ds, np.abs(Es), label=r'$\Delta H$')
    ax.set_title(r'Single step $\Delta H$ computed with OLIVE')
    ax.set_xlabel(r'$1/dt$')
    ax.set_ylabel(r'$\Delta H$')
    ax.set_xlim([1,7e3])
    ax.set_ylim([1e-15,1e-3])
    ax.legend(loc='best')
    #fig.savefig('OLIVE_dH_singlestep.png',bbox_inches='tight')