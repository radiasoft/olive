"""

one_mode is an example file which  seeks to verify the new Hamiltonian algorithm and update sequence
as derived by Dan and Stephen. It will ignore much of the class structure applied to OLIVE in an
attempt to provide a simple example of the algorithm. Once verified, it will be implemented within
OLIVE using the appropriate class structure, memory and data management, I/O, etc. This version uses
a time-dependent Hamiltonian map-based algorithm, as the previous algorithm was unfit to consider vector potentials
with constant geometric dependence along the z-axis.

Nathan Cook
09/08/2016

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
- We must compute frequencies, etc. as desired, and produce an array of mode information corresponding to a
mode index (l) as we see fit. (e.g. l = 0 -> TM mode with m,n,p = 1,1,0 and has corresponding initial Q_l and P_l).

Beginning the algorithm requires transforming the mechanical momentum p to canonical momentum P = p + (e/c)A.


Usage:
------
python one_mode.py
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c
from eigenmodes import calc_A_x, calc_A_y, calc_A_z
from eigenmodes import dx_int_A_z, dy_int_A_z, dz_int_A_z
from eigenmodes import calc_int_A_z
#from eigenmodes import deriv_int_Ax, deriv_int_Ay
#from eigenmodes import OMEGA

# Set the default mass and charge for an electron
m = m_e
q = -100.*e
c = c


class BeamLoader(object):
    """Simple class that simulates particles coupling to cavity fields"""

    NUM_STEPS = 100 #Fix these for now
    NUM_PARTICLES = 2 #Fix these for now

    def __init__(self, q0,p0,Q0,P0,W0,maxTau,MODES,OMEGAS, tau0=0):

        '''
        q0 (ndArray): initial particle positions - dimension (num_particles,3)
        p0 (ndArray): initial particle momenta - dimension (num_particles,3)
        Q0 (ndArray): initial field amplitudes - dimension (num_modes)
        P0 (ndArray): initial amplitude of mode envelope oscillations - dimension (num_modes)
        W0 (ndArray): mode frequencies - dimension (num_modes)

        '''


        #Particles step through tau=ct from tau = 0 to tau = T in N total steps of length h = T/N
        #The first and last (partial) steps have coordinate updates of length 0.5h to set up the initial leap
        #Particles are assumed to have fixed beta, so z velocity is fixed
        #Position at step k is beta*(k-1/2)*h (because position after step 1 is 1/2, after step 2 is 1.5, etc.)
        #self.k = 0 #initial step number


        self.num_steps = BeamLoader.NUM_STEPS
        self.num_particles = BeamLoader.NUM_PARTICLES
        self.num_modes = np.size(np.asarray(Q0))

        #define step size
        self.h = maxTau/self.num_steps
        self.tau = tau0
        #h = self.__h

        #Position quantities
        self.x = q0[:,0]
        self.y = q0[:,1]
        self.z = q0[:,2]

        #Momentum quantities
        self.px = p0[:,0]
        self.py = p0[:,1]
        self.pz = p0[:,2]

        #convert mechanical momentum (P0) to canonical momentum
        self.convert_mechanical_to_canonical()

        #Compute gamma and beta-gamma - note that beta = betagamma/gamma will be held constant
        #p_array = np.einsum('ij,ij->i', p0, p0)

        #self.gamma = np.sqrt((p_array)**2 + (m*c**2)**2) #an extra factor of c already in p
        #self.beta_gamma = p_array/(m*c*c)

        self.gmc = self.calc_gamma_m_c()

        # Field quantities
        self.modes = MODES
        self.omegas = OMEGAS
        self.Q = np.asarray(Q0)
        self.P = np.asarray(P0)
        self.OMEGA = np.asarray(W0)

        #Construct history arrays to store updates
        self.x_history = [self.x]
        self.y_history = [self.y]
        self.z_history = [self.z]
        self.px_history = [self.px]
        self.py_history = [self.py]
        self.pz_history = [self.pz]
        self.Q_history = [self.Q]
        self.P_history = [self.P]
        self.tau_history = [tau0]



    def convert_mechanical_to_canonical(self):
        '''Convert mechanical momenta to canonical momenta for the current particle state'''

        A_x = calc_A_x(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_y = calc_A_y(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_z = calc_A_z(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)


        self.px = self.px + (e/c) * np.dot(self.Q,A_x)
        self.py = self.py + (e / c) * np.dot(self.Q,A_y)
        self.pz = self.pz + (e / c) * np.dot(self.Q,A_z)

    def convert_canonical_to_mechanical(self):
        '''Convert mechanical momenta to canonical momenta for the current particle state'''

        A_x = calc_A_x(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_y = calc_A_y(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_z = calc_A_z(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)

        self.px = self.px - (e/c) * np.dot(self.Q,A_x)
        self.py = self.py - (e / c) * np.dot(self.Q,A_y)
        self.pz = self.pz - (e / c) * np.dot(self.Q,A_z)


    def compute_spatial_eigenmode(self, x,y,z):
        '''Return an array of spatial eigenmode vectors at position (x,y,z)
        Here we've assumed that some function f_mode(x,y,z) has been imported from the
        external file 'eigenmodes.py' which computes an array of spatial eigenmode
        values in the x,y,and z directions for each mode.

        Arguments:
            x,y,z (ndarray): arrays of particle positions

        Returns:
            fz, fz_dx, fz_dy, fz_dz (ndarray): arrays of eigenmode evaluations and
                                               spatial derivatives - each of length num_modes

        '''

        return f_mode(x,y,z)


    def calc_gamma_m_c(self):
        '''Compute the quantity gamma*m*c for every particle and update the corresponding member variable'''

        A_x = calc_A_x(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_y = calc_A_y(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)
        A_z = calc_A_z(self.modes[:, 0], self.modes[:, 1], self.modes[:, 2], self.x, self.y, self.z)

        self.gmc = np.sqrt( (self.px - (e/c)*np.dot(self.Q,A_x))**2 + (self.py- (e/c)*np.dot(self.Q,A_y))**2 + (self.py - (e/c)*np.dot(self.Q,A_z))**2 + (m*c)**2)


    def update_q(self, k=0,step=1.):
        '''Update for all particles a single component qk -> qk+1(or 1/2) given the momentum pk

        Arguments:
            k (int): Index of coordinate being advance (k=0 for x, k=1 for y, k=2 for z)
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''
        if k ==0:
            self.x = self.x + step* self.h * self.px / self.gmc
        elif k == 1:
            self.y = self.y + step* self.h * self.py / self.gmc
        elif k == 2:
            self.z = self.z + step* self.h * self.pz / self.gmc
        else:
            raise ValueError("Coordinate index outside of range [0,1,2]")

    def kick_p(self, k=0,sign=1, step=1.):
        '''Kick p is the kick portion of the coupling Hamiltonian map, which updates each component of p as well
         as the field momentum P. The kick remains dependent upon the coordinate subscript k, as it differs for each
         coordinate-specific map x,y,z.

        Arguments:
            k (int): Index of coordinate dictating the advance (k=0 for x, k=1 for y, k=2 for z)
            sign (int): 1 if subtracting (e.g. first step), -1 if adding (e.g. 2nd step)
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        if k ==0:
            #k = 0 means evaluating A_x for all field couplings
            # Each function returns an LxN array - integral values for L modes evaluated at N particle positions.
            ddx_int_A_x = np.zeros((self.num_modes,self.num_steps))
            ddy_int_A_x = np.zeros((self.num_modes,self.num_steps))
            ddz_int_A_x = np.zeros((self.num_modes,self.num_steps))

            # LxN array->int_A_x evaluate for L modes at N particle positions
            int_A_x = np.zeros((self.num_modes,self.num_steps))

            # np.dot(self.Q,ddy_int_A_x) is the same as np.einsum('i,ij->j',Q,dxintA_L)

            self.px = self.px - sign*step*self.h*(e/c)*np.dot(self.Q,ddx_int_A_x)
            self.py = self.py - sign*step*self.h*(e/c)*np.dot(self.Q,ddy_int_A_x)
            self.pz = self.pz - sign*step*self.h*(e/c)*np.dot(self.Q,ddz_int_A_x)

            #Update modementa
            # sum over each particle for all l modes -> array of length l
            self.P = self.P - sign*step*self.h*(e/c)*np.einsum('ij->i', int_A_x)

        elif k == 1:
            #k = 1 means evaluating A_y for all field couplings
            ddx_int_A_y = np.zeros((self.num_modes,self.num_steps))
            ddy_int_A_y = np.zeros((self.num_modes,self.num_steps))
            ddz_int_A_y = np.zeros((self.num_modes,self.num_steps))

            # LxN array->int_A_y evaluate for L modes at N particle positions
            int_A_y = np.zeros((self.num_modes,self.num_steps))

            self.px = self.px - sign*step*self.h*(e/c)*np.dot(self.Q,ddx_int_A_y) #self.Q*ddx_int_A_y
            self.py = self.py - sign*step*self.h*(e/c)*np.dot(self.Q,ddy_int_A_y) #self.Q*ddy_int_A_y
            self.pz = self.pz - sign*step*self.h*(e/c)*np.dot(self.Q,ddz_int_A_y) #self.Q*ddz_int_A_y

            #Update modementa
            contribution = 1. #array of L x N (L modes and N particles)
            # sum over each particle for all l modes -> array of length l
            self.P = self.P - sign*step*self.h*(e/c)*np.einsum('ij->i',int_A_y)


        elif k == 2:
            #k = 2 means evaluating A_z for all field couplings
            # Returns an LxN array - integral values for L modes evaluated at N particle positions.
            ddx_int_A_z = dx_int_A_z(self.modes[:,0],self.modes[:,1],self.modes[:,2],self.x,self.y,self.z)
            ddy_int_A_z = dy_int_A_z(self.modes[:,0],self.modes[:,1],self.modes[:,2],self.x,self.y,self.z)
            ddz_int_A_z = dz_int_A_z(self.modes[:,0],self.modes[:,1],self.modes[:,2],self.x,self.y,self.z)

            # LxN array->int_A_z evaluate for L modes at N particle positions
            int_A_z = calc_int_A_z(self.modes[:,0],self.modes[:,1],self.modes[:,2],self.x,self.y,self.z)

            self.px = self.px - sign*step*self.h*(e/c)*np.dot(self.Q,ddx_int_A_z) #self.Q*ddx_int_A_z
            self.py = self.py - sign*step*self.h*(e/c)*np.dot(self.Q,ddy_int_A_z) #self.Q*ddy_int_A_z
            self.pz = self.pz - sign*step*self.h*(e/c)*np.dot(self.Q,ddz_int_A_z) #self.Q*ddz_int_A_z

            #Update modementa
            contribution = 1. #array of L x N (L modes and N particles)
            # sum over each particle for all l modes -> array of length l
            self.P = self.P - sign*step*self.h*(e/c)*np.einsum('ij->i',int_A_z)

        else:
            raise ValueError("Coordinate index outside of range [0,1,2]")

    def update_x(self, step=1.):
        '''Perform the map for x, consisting of a half kick, drift, then half kick'''

        self.kick_p(k=0,sign=1,step=0.5*step)
        self.update_q(k=0,step=step)
        self.kick_p(k=0,sign=-1,step=0.5*step) #reverse the sign for the 2nd kick due to similarity transform

    def update_y(self, step=1.):
        '''Perform the map for y consisting of a half kick, drift, then half kick'''

        self.kick_p(k=1,sign=1,step=0.5*step)
        self.update_q(k=1,step=step)
        self.kick_p(k=1,sign=-1,step=0.5*step) #reverse the sign for the 2nd kick due to similarity transform

    def update_z(self, step=1.):
        '''Perform the map for y consisting of a half kick, drift, then half kick'''

        self.kick_p(k=2,sign=1,step=0.5*step)
        self.update_q(k=2,step=step)
        self.kick_p(k=2,sign=-1,step=0.5*step) #reverse the sign for the 2nd kick due to similarity transform


    def rotate_fields(self, step=1.):
        '''Update field phases self consistently with the time step.

        Arguments:
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        Note that this step applies a fixed rotation operator that only varies on the size
        of the time step.

        '''

        currentQ = self.Q
        currentP = self.P

        self.P = currentP*np.cos(self.OMEGA * step * self.h) - (1./self.OMEGA)*currentQ*np.sin(self.OMEGA* step * self.h)

        self.Q = currentP *self.OMEGA* np.sin(self.OMEGA * step * self.h) + currentQ * np.cos(self.OMEGA * step * self.h)



    def update_histories(self):
        '''Updates all coordinate histories with current values'''

        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.z_history.append(self.z)
        self.px_history.append(self.px)
        self.py_history.append(self.py)
        self.pz_history.append(self.pz)

        self.Q_history.append(self.Q)
        self.P_history.append(self.P)


    def step(self, N=1):
        '''Perform N steps in z'''

        print "Beginning step 1"

        # Initial half step field rotation
        self.rotate_fields(step=0.5)


        #print len(self.Q)

        #update times for diagnostics and momenta for consistency
        #self.tau = self.tau + 0.5*self.h
        #self.tau_history.append(self.tau)
        #self.px_history.append(self.px)
        #self.py_history.append(self.py)
        #self.pz_history.append(self.pz)

        i = N-0.5

        while i > 1:
            self.calc_gamma_m_c()
            self.update_x(step=0.5)
            self.update_y(step=0.5)
            self.update_z(step=1.)
            self.update_y(step=0.5)
            self.update_x(step=0.5)

            self.rotate_fields(step=1)

            # update times for diagnostics
            self.tau = self.tau + self.h
            self.tau_history.append(self.tau)

            #update coordinate histories
            self.update_histories()

            i = i - 1
            #print len(self.Q)
            #print "New z coordinate is {}".format(self.z)

        #To finish, do our full x,y,z update sequence and a final half-step rotation of fields
        self.calc_gamma_m_c()
        self.update_x(step=0.5)
        self.update_y(step=0.5)
        self.update_z(step=1.)
        self.update_y(step=0.5)
        self.update_x(step=0.5)

        self.rotate_fields(step=0.5)

        # update times for diagnostics - and moment for consistency
        self.tau = self.tau + 0.5*self.h
        self.tau_history.append(self.tau)

        self.update_histories()







    def plot_coordinates(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12,8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history)/c,np.asarray(self.z_history)[:,0], label = 'z')
        ax.plot(np.asarray(self.tau_history)/c,np.asarray(self.x_history)[:,0], label = 'x')
        #ax.plot(np.asarray(self.tau_history)/c,self.y_history[:,0], label = 'y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('Particle in mode 110')
        ax.legend(loc='best')
        plt.savefig('coordinates_110.pdf',bbox_inches='tight')

    def plot_momenta(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12,8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history)/c,np.asarray(self.pz_history)[:,0]/(m_e*c**2), label = 'z')
        #ax.plot(np.asarray(self.tau_history)/c,np.asarray(self.px_history)[:,0]/(m_e*c**2), label = 'x')
        #ax.plot(np.asarray(self.tau_history)/c,self.py_history[:,0], label = 'y')

        ax.set_ylim([999.5,1000.5])

        ax.set_xlabel('Time')
        ax.set_ylabel('Momenta')
        ax.set_title('Particle in mode 110')
        ax.legend(loc='best')
        plt.savefig('momenta_110.pdf',bbox_inches='tight')

    def plot_field_amplitudes(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12,8))

        ax = fig.gca()

        new_Qs = np.asarray([Q[0] for Q in self.Q_history])

        ax.plot(np.asarray(self.tau_history)/c,new_Qs, label = r'$a_0$')
        #ax.plot(self.tau_history,self.x_history, label = 'x')
        #ax.plot(self.tau_history,self.y_history, label = 'y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Fields in mode 110')
        ax.legend(loc='best')
        plt.savefig('amplitudes_110.pdf',bbox_inches='tight')


        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        new_Ps = np.asarray([P[0] for P in bload.P_history])
        leg_Ps = np.concatenate(([new_Ps[0]], new_Ps[::2]))
        ax.plot(np.asarray(self.tau_history) / c, leg_Ps, label=r'$a_0$')
        # ax.plot(self.tau_history,self.x_history, label = 'x')
        # ax.plot(self.tau_history,self.y_history, label = 'y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Fields in mode 110')
        ax.legend(loc='best')
        plt.savefig('oscillations_110.pdf', bbox_inches='tight')




if __name__ == '__main__':

    a = 0.1
    b = 0.1
    d = 0.2


    #starting coordinates
    q1 = [a / 2., b / 2., 0.]
    q2 = [a / 2., b / 4., 0.]
    q3 = [a / 4., b / 2., 0.]
    q4 = [a / 4., b / 4., 0.]
    #q2 = [a / 2.5, b / 2.5, 0.]
    q0 = np.asarray([q1, q1])
    #q0 =np.asarray([a / 2., b / 2.,0.])

    gamma = 100  # 50 MeV electrons

    # To properly normalize, define angles of trjectories
    frac_z = 0.95  # fraction of total energy in z-plane
    frac_y_x = 0.5  # fraction of remaining energy in y-plane
    angle = np.arccos(frac_y_x)  # corresponding anglae

    left = np.sqrt(1 - frac_z ** 2)
    frac_y = np.cos(angle) * left
    frac_x = np.sin(angle) * left

    #unitless = [frac_x, frac_y, frac_z]
    unitless = [0.,0.,1.]

    frac_z = 0.9  # fraction of total energy in z-plane
    frac_y_x = 0.5  # fraction of remaining energy in y-plane
    angle = np.arccos(frac_y_x)  # corresponding angle

    left = np.sqrt(1 - frac_z ** 2)
    frac_y = np.cos(angle) * left
    frac_x = np.sin(angle) * left

    unitless2 = [frac_x, frac_y, frac_z]

    # Our canonical momentum is actually an energy (e.g. p*c)
    p1 = gamma * m_e * c * np.asarray(unitless) * c
    #p2 = gamma * m_e * c * np.asarray(unitless2) * c
    #p0 = np.asarray([p2, p2])
    p0 = np.asarray([p1,p1])

    # Our canonical momentum is actually an energy (e.g. p*c)
    #p1 = gamma * m_e * c*np.asarray([0., 0.01, 0.99])*c
    #p2 = gamma * m_e * c*np.asarray([0., 0.01, 0.99])*c
    #p0 = np.asarray([p1, p2])

    #Mode values - only 1 mode here
    Q0 = [1.e1] #normalize the amplitude here to 1...?
    P0 = [0.] #this is essentially set by the phase (e.g. t = 0) -> let's assume we're on phase
    W0 = [OMEGA/c] #imported from eigenmodes.py - NOTE THAT WE ARE DIVIDING OUT A FACTOR OF C HERE DUE TO TIMESTEP


    #Define max time via fundamental frequency - it will be one oscillation of the fields
    #num_osc = .1
    #maxT = num_osc*(2*np.pi)/(W0[0]*c)
    #maxTau = c*maxT
    maxTau = d #traverse one cavity length

    # Integrate the equations of motion with a timestep dt
    bload = BeamLoader(q0,p0,Q0,P0,W0,maxTau)
    bload.step(bload.num_steps)


    bload.plot_coordinates()
    bload.plot_momenta()
    #print len(bload.tau_history)
    #print len(bload.Q_history)
    #print bload.Q_history
    bload.plot_field_amplitudes()
    #print len(bload.tau_history)

    #print bload.px_history
    #new_Ps = np.asarray([P[0] for P in bload.P_history])
    #leg_Ps = np.concatenate(([new_Ps[0]],new_Ps[::2]))
    #print len(leg_Ps)


    print np.asarray(bload.pz_history)[::10,0]

    #print len([P[0] for P in bload.P_history])