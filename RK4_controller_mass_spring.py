# -*- coding: utf-8 -*-
"""
@author: João Angelo Ferres Brogin

São Paulo State University (UNESP) - School of Engineering of Ilha Solteira (FEIS)
Group of Intelligent Materials and Systems (GMSINT)
Post-Graduation Program in Mechanical Engineering (PPGEM)

The following code was implemented with the purpose of designing and applying a fuzzy observer to estimate the unmeasurable variables
of the dynamic model known as Epileptor (Jirsa et al., 2014), which represents the main features of epileptiform activity. 
In this file, the dynamic matrix for a mass-spring-damper system is defined to study its behavior when a controller using decay rates is applied.
The main objective is to compare the classic approach using the parameter alpha (Tanaka & Wang, 2004) with the approach proposed in the paper using beta, in terms of reducing the time necessary for the system to achieve the stability poitn (0,0).

The main results obtained can be found in the article entitled (Appendix D): 
Brogin, J. A. F., Faber, J. and Bueno, D. D. "Burster Reconstruction Considering Unmeasurable Variables in the Epileptor Model".

Plase cite this code as: 
Brogin, J. A. F., Faber, J. and Bueno, D. D. (2021) Burster reconstruction considering unmeasurable variables in the Epileptor model [Computer code]. 

References:
[1] Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. Brain, 137(8), 2210-2230.
[2] Tanaka, K., & Wang, H. O. (2004). Fuzzy control systems design and analysis: a linear matrix inequality approach. John Wiley & Sons.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from LMIs_controller_mass_spring import P, G

#%% Simulation parameters:
Fs = 100
dt = 1/Fs
nd = 2        
N = 2000    
X = np.zeros((nd, N))
t0 = 0
tf = N*dt
t = np.linspace(t0, tf, N)

#%% Parameters of the model:
m = 1
K = 1
c = 0

#%% Predefined vector for the input forces:
U = [None]*N

#%% Differential equations in state-space notation:
def __dXdt__( Xd, switch ):
    x = Xd[0]
    y = Xd[1]

    X_ = np.array([x, y])
    
    # Dynamic Matrix A:
    A = np.array([  [      0,     1   ], \
                    [   -K/m,   -c/m   ] ]) 
        
    # Matrix B:
    B = np.array([ 0, \
                  1/(m)])  
    
    # Solution:
    aux = A.dot(X_)
    
    UT = -switch*G.dot(X_)
    wu = np.reshape(B, (-1,1)).dot(UT.T)
       
    # Solution with the application of the controller:
    sol = np.reshape(aux, (-1,1)) + wu
    sol = np.array([ float(sol[0]), float(sol[1]) ])
    
    return sol, UT
    
#%% Fourth-Order Runge-Kutta Integrator:     
V = np.zeros(N)

# Initial conditions: to obtain more solutions, change the values of the 'for' loop
for ii in range(0,1):
    x0 = 0
    y0 = 0.1
    X[:,0] = [x0, y0]
    
    for k in range(0, N-1):
        # If switch = 1, the controller is turned on. If switch = 0, no control is applied
        switch = 1 
        print('Iteration number: ' + str(k))
        k1, u = __dXdt__( X[:,k], switch )
        k2, u = __dXdt__( X[:,k] + k1*(dt/2), switch )
        k3, u = __dXdt__( X[:,k] + k2*(dt/2), switch )
        k4, u = __dXdt__( X[:,k] + k3*dt, switch )
        X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) 

        # Lyapunov function:
        V[k] =  float(np.reshape(X[:,k], (-1,1)).T * P * np.reshape(X[:,k], (-1,1))) 
                              
        # Forces:
        U[k] = u

#%% Response in the time domain:
plt.figure(1)
plt.subplot(211)
plt.plot(t, X[0,:], 'b', linewidth=4)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$x(t)$', fontsize=15)
plt.xlim(t[0], 10)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(212)
plt.plot(t, X[1,:], 'k', linewidth=4)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$y(t)$', fontsize=15)
plt.xlim(t[0], 10)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Lyapunov function:
plt.figure(2)
plt.subplot(211)
plt.plot(t, V, 'k', linewidth = 4)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$V(\mathbf{x}(t))$', fontsize=15)
plt.grid()
plt.xlim(t[0], 10)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(212)
plt.plot(t, np.gradient(V), 'r', linewidth = 4)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$\dot{V}(\mathbf{x}(t))$', fontsize=15)
plt.grid()
plt.xlim(t[0], 10)
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Phase portrait:
plt.figure(4)
plt.plot(X[0,:], X[1,:], 'k', linewidth=4)
plt.xlabel('$x$', fontsize=15, rotation=0)
plt.ylabel('$y$', fontsize=15, rotation=0)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

#%% 3D plot to visualize the convergence:
plt.figure(5)
ax = plt.axes(projection='3d')
ax.plot3D(X[0,:], X[1,:], V, 'b', linewidth=2)
plt.xlabel('$x$', fontsize=15, rotation=0)
plt.ylabel('$y$', fontsize=15, rotation=0)
ax.set_zlabel(r'$V(x,y)$', fontsize=15, rotation=0)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.grid()

plt.figure(5)
plt.plot(np.zeros(len(V)),  X[1,:], 'k--', linewidth=2)
plt.plot(X[0,:], np.zeros(len(V)), 'k--', linewidth=2)