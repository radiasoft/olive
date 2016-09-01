"""

new_flat_run is an example file which  seeks to verify the initial algorithm and update sequence
as derived by Dan and Stephen. It will ignore much of the class structure applied to OLIVE in an
attempt to provide a simple example of the algorithm. Once verified, it will be implemented within
OLIVE using the appropriate class structure, memory and data management, I/O, etc. This version uses
the time dependent algorithm rather than the z-dependent algorithm.

Nathan Cook
08/30/2016

For starters, just consider a single particle?

Usage:
------
python new_flat_run.py
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c
from eigenmodes import f_mode, OMEGA

# Set the default mass and charge for an electron
m = m_e
q = -e


class BeamLoader(object):
    """Simple class that simulates particles coupling to cavity fields"""

    NUM_STEPS = 1000 #Fix these for now
    NUM_PARTICLES = 1 #Fix these for now

    def __init__(self, q0,p0,Q0,P0,W0,maxTau, tau0=0):

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

        #Compute gamma and beta-gamma - note that beta = betagamma/gamma will be held constant
        p_array = np.einsum('ij,ij->i', p0, p0)

        self.gamma = np.sqrt((p_array*c)**2 + (m*c**2)**2)
        self.beta_gamma = p_array/(m*c)


        # Field quantities
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


    def update_beta_gamma(self):
        '''Compute beta_gamma for the k+1/2th step given velocities at the kth step'''

        #self.beta_gamma = np.sqrt(self.ux**2 + self.uy**2 - self.uz**2)

        #self.u_history.append(self.beta_gamma)

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

    def update_qs(self, step=1.):
        '''Update qk -> qk+1(or 1/2) given the momentum pk

        Arguments:
            step (Float): Fraction of a step to advance the coordinates (usually 0.5 or 1)

        '''

        self.x = self.x + step* self.h * self.px / (np.sqrt(m**2*c**2*(1+self.beta_gamma)))
        self.y = self.y + step* self.h * self.py / (np.sqrt(m**2*c**2*(1+self.beta_gamma)))
        self.z = self.z + step* self.h * self.pz / (np.sqrt(m**2*c**2*(1+self.beta_gamma)))

        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.z_history.append(self.z)

    def update_ps(self):
        '''Update velocities for k+1/2th step given velocities at kth step and positions at k+1/2 step

        NOTE: This update only occurs from k -> k+1/2. Continuity requires u(k+1) = u(k+1/2), so no
        update is required for that step.

        '''

        #Compute spatial eigenmodes and derivatives for each particle
        fz, f_dx, f_dy, f_dz = f_mode(self.x,self.y,self.z)

        #Assume only one mode for now!
        self.px = self.px + self.h*q*(self.beta_gamma/self.gamma)*self.Q*f_dx
        self.py = self.py + self.h*q*(self.beta_gamma/self.gamma)*self.Q*f_dy
        self.pz = self.pz + self.h*q*(self.beta_gamma/self.gamma)*self.Q*f_dz

        self.px_history.append(self.px)
        self.py_history.append(self.py)
        self.pz_history.append(self.pz)

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


        self.Q_history.append(self.Q)
        self.P_history.append(self.P)

    def update_Ps(self):
        '''Update mode amplitudes Pk -> Pk+1 given qk+1/2'''

        # Compute spatial eigenmodes and derivatives for each particle
        fz, f_dx, f_dy, f_dz = f_mode(self.x, self.y, self.z)

        # Assume only one mode for now!
        self.P = self.P + self.h * q * (self.beta_gamma / self.gamma) * np.sum(fz)

        self.P_history.append(self.P)




    def step(self, N=1):
        '''Perform N steps in z'''

        print "Beginning step 1"

        # Initial half step update of q, and field rotation
        self.update_qs(step=0.5)
        self.rotate_fields(step=0.5)

        #print len(self.Q)

        #update times for diagnostics and momenta for consistency
        self.tau = self.tau + 0.5*self.h
        self.tau_history.append(self.tau)
        self.px_history.append(self.px)
        self.py_history.append(self.py)
        self.pz_history.append(self.pz)

        i = N-0.5

        while i > 1:
            self.update_ps()
            self.update_Ps()
            self.update_qs()
            self.rotate_fields()

            # update times for diagnostics
            self.tau = self.tau + self.h
            self.tau_history.append(self.tau)

            i = i - 1

            #print len(self.Q)
            #print "New z coordinate is {}".format(self.z)

        #To finish, do a full step in p and P, half step in q, and rotate
        self.update_ps()
        self.update_Ps()
        self.update_qs(step=0.5)
        self.rotate_fields(step=0.5)

        # update times for diagnostics - and moment for consistency
        self.tau = self.tau + 0.5*self.h
        self.tau_history.append(self.tau)
        #self.px_history.append(self.px)
        #self.py_history.append(self.py)
        #self.pz_history.append(self.pz)



    def plot_coordinates(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12,8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history)/c,self.z_history, label = 'z')
        ax.plot(np.asarray(self.tau_history)/c,self.x_history, label = 'x')
        ax.plot(np.asarray(self.tau_history)/c,self.y_history, label = 'y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('Particle in mode 110')
        ax.legend(loc='best')
        plt.savefig('coordinates_110.pdf',bbox_inches='tight')

    def plot_momenta(self):
        '''Plot the particle coordinates history'''

        fig = plt.figure(figsize=(12,8))

        ax = fig.gca()
        ax.plot(np.asarray(self.tau_history)/c,self.pz_history, label = 'z')
        ax.plot(np.asarray(self.tau_history)/c,self.px_history, label = 'x')
        ax.plot(np.asarray(self.tau_history)/c,self.py_history, label = 'y')

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
    q2 = [a / 2.5, b / 2.5, 0.]
    q0 = np.asarray([q1, q2])

    gamma = 1000  # 500 MeV electrons
    # Assume just z momentum here
    p1 = gamma * m_e * c*np.asarray([0., 0.01, 0.99])
    p2 = gamma * m_e * c*np.asarray([0., 0.01, 0.99])
    p0 = np.asarray([p1, p2])

    #Mode values - only 1 mode here
    Q0 = [2.e3] #normalize the amplitude here to 1...?
    P0 = [0.] #this is essentially set by the phase (e.g. t = 0) -> let's assume we're on phase
    W0 = [OMEGA] #imported from eigenmodes.py


    #Define max time via fundamental frequency - it will be one oscillation of the fields
    num_osc = 1.
    maxT = num_osc*(2*np.pi)/W0[0]
    maxTau = c*maxT

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

    #new_Ps = np.asarray([P[0] for P in bload.P_history])
    #leg_Ps = np.concatenate(([new_Ps[0]],new_Ps[::2]))
    #print len(leg_Ps)

    #print len([P[0] for P in bload.P_history])