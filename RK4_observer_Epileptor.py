# -*- coding: utf-8 -*-
"""
@author: João Angelo Ferres Brogin

São Paulo State University (UNESP) - School of Engineering of Ilha Solteira (FEIS)
Group of Intelligent Materials and Systems (GMSINT)
Post-Graduation Program in Mechanical Engineering (PPGEM)

The following code was implemented with the purpose of designing and applying a fuzzy observer to estimate the unmeasurable variables
of the dynamic model known as Epileptor (Jirsa et al., 2014), which represents the main features of epileptiform activity. 
In this file, the dynamic matrices for both model and observer are defined, and a Fourth-Order Runge-Kutta integrator is run to obtain both the estimated and actual states.

The main results obtained can be found in the article entitled: 
Brogin, J. A. F., Faber, J. and Bueno, D. D. "Burster Reconstruction Considering Unmeasurable Variables in the Epileptor Model".

Plase cite this code as: 
Brogin, J. A. F., Faber, J. and Bueno, D. D. (2021) Burster reconstruction considering unmeasurable variables in the Epileptor model: Fourth-Order Runge-Kutta integrator [Computer code]. 

References:
[1] Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. Brain, 137(8), 2210-2230.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from LMIs_observer_Epileptor import C, L, A_base, zeta1_max, zeta1_min, zeta2_max, zeta2_min, zeta3_max, zeta3_min, \
                                      zeta4_max, zeta4_min, zeta5_max, zeta5_min, zeta6_max, zeta6_min
                                                
#%% Simulation parameters:
Fs = 100
dt = 1/Fs
nd = 6        
N = 200000 # 80000  
t0 = 0
tf = N*dt
t = np.linspace(t0, tf, N)
X = np.zeros((nd, N))
Xc = np.zeros((nd, N))
E = np.zeros((nd, N))

#%% Parameters of he model:
tau1 = 1
tau2 = 10   
tau0 = 2857.0
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Predefined variables for the nonlinear functions and membership functions:
der1 = np.zeros(N)
der2 = np.zeros(N)
der3 = np.zeros(N)
der4 = np.zeros(N)
der5 = np.zeros(N)
der6 = np.zeros(N)

mu1_mf = np.zeros(N)
mu2_mf = np.zeros(N)
gamma1_mf = np.zeros(N)
gamma2_mf = np.zeros(N)
alpha1_mf = np.zeros(N)
alpha2_mf = np.zeros(N)
omega1_mf = np.zeros(N)
omega2_mf = np.zeros(N)
beta1_mf = np.zeros(N)
beta2_mf = np.zeros(N)
epsilon1_mf = np.zeros(N)
epsilon2_mf = np.zeros(N)

#%% Differential equations in state-space notation:
def __dXdt__( Xd, Xd_c, Ed ):
    global h1, h1z, c1, c3, h2, c2, h1c, h1zc, c1c, c3c, h2c, c2c, d1c, d3c, d6c, d1, d3, d6
    # States of the model/plant:
    x1 = Xd[0]
    y1 = Xd[1]
    z = Xd[2]
    x2 = Xd[3]
    y2 = Xd[4]
    u = Xd[5]
    X_ = np.array([x1,y1,z,x2,y2,u])
    
    # States of the observer:
    x1c = Xd_c[0]
    y1c = Xd_c[1]
    zc = Xd_c[2]
    x2c = Xd_c[3]
    y2c = Xd_c[4]
    uc = Xd_c[5]
    X_c = np.array([x1c,y1c,zc,x2c,y2c,uc])

    # Estimation errors:
    e1 = Ed[0]
    e2 = Ed[1]
    e3 = Ed[2]
    e4 = Ed[3]
    e5 = Ed[4]
    e6 = Ed[5]
    E_ = np.array([e1,e2,e3,e4,e5,e6])
    
    ############################## MODEL #####################################
    
    if x1 < 0:
        h1 = x1**2 - 3*x1
        d1 = -3*x1**2 + 6*x1 
        d3 = -1 
    elif x1 >= 0:
        h1 = x2 - 0.6*z**2 + 4.8*z - 9.6
        d1 = -x2 + 0.6*(z - 4)**2 
        d3 = 1.2*x1*z - 4.8*x1 - 1 
    if x2 < -0.25:
        h2 = 0
        c3 = 0
        d6 = 0 
    elif x2 >= -0.25:
        h2 = 6
        c3 = 1.5
        d6 = 6/tau2 
        
    ############################ OBSERVER ###################################
    
    if x1c < 0:
        h1c = x1c**2 - 3*x1c
        d1c = -3*x1c**2 + 6*x1c 
        d3c = -1 
    elif x1c >= 0:
        h1c = x2c - 0.6*zc**2 + 4.8*zc - 9.6
        d1c = -x2c + 0.6*(zc - 4)**2 
        d3c = 1.2*x1c*zc - 4.8*x1c - 1 
    if x2c < -0.25:
        h2c = 0
        c3c = 0
        d6c = 0 
    elif x2c >= -0.25:
        h2c = 6
        c3c = 1.5
        d6c = 6/tau2 
        
    ##################### MEMBERSHIP FUNCTIONS ##############################
    
    d2 = -10*x1/tau1 
    d4 = -x1 
    d5 = 1 - 3*x2**2 
    
    d2c = -10*x1c/tau1
    d4c = -x1c
    d5c = 1 - 3*x2c**2
    
    D = [d1, d2, d3, d4, d5, d6]
    
    mu1 = (zeta1_max - d1c)/(zeta1_max - zeta1_min)
    mu2 = 1 - mu1

    gamma1 = (zeta2_max - d2c)/(zeta2_max - zeta2_min)
    gamma2 = 1 - gamma1
    
    alpha1 = (zeta3_max - d3c)/(zeta3_max - zeta3_min)
    alpha2 = 1 - alpha1
    
    omega1 = (zeta4_max - d4c)/(zeta4_max - zeta4_min)
    omega2 = 1 - omega1
    
    beta1 = (zeta5_max - d5c)/(zeta5_max - zeta5_min)
    beta2 = 1 - beta1
    
    epsilon1 = (zeta6_max - d6c)/(zeta6_max - zeta6_min)
    epsilon2 = 1 - epsilon1
    
    memb_func = [mu1,mu2,gamma1,gamma2,alpha1,alpha2,omega1,omega2,beta1,beta2,epsilon1,epsilon2]
    
    ###########################################################################
    
    # Matrix A (model):
    A = np.array([  [        -h1,         1,          -1,               0,            0,            0], \
                    [      -5*x1,        -1,           0,               0,            0,            0], \
                    [     4/tau0,         0,     -1/tau0,               0,            0,            0], \
                    [          0,         0,        -0.3,    ( 1 - x2**2),           -1,            2], \
                    [          0,         0,           0,         h2/tau2,      -1/tau2,            0], \
                    [  0.1*gamma,         0,           0,               0,            0,       -gamma] ]) 
    
    # Matrix A (observer):
    Ac = np.array([ [       -h1c,         1,          -1,               0,            0,            0], \
                    [     -5*x1c,        -1,           0,               0,            0,            0], \
                    [     4/tau0,         0,     -1/tau0,               0,            0,            0], \
                    [          0,         0,        -0.3,   ( 1 - x2c**2),           -1,            2], \
                    [          0,         0,           0,        h2c/tau2,      -1/tau2,            0], \
                    [  0.1*gamma,         0,           0,               0,            0,       -gamma] ])   
    
    # Matrix B (model):
    b = np.array([ I1 , \
                  y0 , \
                  - 4*x0/tau0 , \
                  I2 + (3.5*0.3), \
                  c3/tau2, \
                  0])    
   
    # Matrix B (observer):
    bc = np.array([ I1 , \
                   y0 , \
                   - 4*x0/tau0, \
                   I2 + (3.5*0.3), \
                   c3c/tau2, \
                   0])       
    
    # Each variable is obtained separately:
    aux = A.dot(X_)
    t1 = float(aux[0] + b[0])
    t2 = float(aux[1] + b[1])
    t3 = float(aux[2] + b[2])
    t4 = float(aux[3] + b[3])
    t5 = float(aux[4] + b[4])
    t6 = float(aux[5] + b[5])
    sol_p = np.array([t1, t2, t3, t4, t5, t6])
    
    p1 = Ac.dot(X_c)
    y_tilde = np.reshape(X_ - X_c,(6,-1))
    scaling_factor = 1     
    p = scaling_factor*L*C*y_tilde  

    t1_c = float(p1[0] + bc[0] + p[0])
    t2_c = float(p1[1] + bc[1] + p[1])
    t3_c = float(p1[2] + bc[2] + p[2])
    t4_c = float(p1[3] + bc[3] + p[3])
    t5_c = float(p1[4] + bc[4] + p[4])
    t6_c = float(p1[5] + bc[5] + p[5])
    sol_c = np.array([t1_c,t2_c,t3_c,t4_c,t5_c,t6_c])
       
    p2 =  mu2*gamma2*alpha2*omega2*beta2*epsilon2*(A_base[0] )
    p3 =  mu2*gamma2*alpha2*omega2*beta2*epsilon1*(A_base[1] )
    p4 =  mu2*gamma2*alpha2*omega2*beta1*epsilon2*(A_base[2] )
    p5 =  mu2*gamma2*alpha2*omega2*beta1*epsilon1*(A_base[3] )
    p6 =  mu2*gamma2*alpha2*omega1*beta2*epsilon2*(A_base[4] )
    p7 =  mu2*gamma2*alpha2*omega1*beta2*epsilon1*(A_base[5] )
    p8 =  mu2*gamma2*alpha2*omega1*beta1*epsilon2*(A_base[6] )
    p9 =  mu2*gamma2*alpha2*omega1*beta1*epsilon1*(A_base[7] )
    p10 = mu2*gamma2*alpha1*omega2*beta2*epsilon2*(A_base[8] )
    p11 = mu2*gamma2*alpha1*omega2*beta2*epsilon1*(A_base[9] )
    p12 = mu2*gamma2*alpha1*omega2*beta1*epsilon2*(A_base[10])
    p13 = mu2*gamma2*alpha1*omega2*beta1*epsilon1*(A_base[11])
    p14 = mu2*gamma2*alpha1*omega1*beta2*epsilon2*(A_base[12])
    p15 = mu2*gamma2*alpha1*omega1*beta2*epsilon1*(A_base[13])
    p16 = mu2*gamma2*alpha1*omega1*beta1*epsilon2*(A_base[14])
    p17 = mu2*gamma2*alpha1*omega1*beta1*epsilon1*(A_base[15])
    p18 = mu2*gamma1*alpha2*omega2*beta2*epsilon2*(A_base[16])
    p19 = mu2*gamma1*alpha2*omega2*beta2*epsilon1*(A_base[17])
    p20 = mu2*gamma1*alpha2*omega2*beta1*epsilon2*(A_base[18])
    p21 = mu2*gamma1*alpha2*omega2*beta1*epsilon1*(A_base[19])
    p22 = mu2*gamma1*alpha2*omega1*beta2*epsilon2*(A_base[20])
    p23 = mu2*gamma1*alpha2*omega1*beta2*epsilon1*(A_base[21])
    p24 = mu2*gamma1*alpha2*omega1*beta1*epsilon2*(A_base[22])
    p25 = mu2*gamma1*alpha2*omega1*beta1*epsilon1*(A_base[23])
    p26 = mu2*gamma1*alpha1*omega2*beta2*epsilon2*(A_base[24])
    p27 = mu2*gamma1*alpha1*omega2*beta2*epsilon1*(A_base[25])
    p28 = mu2*gamma1*alpha1*omega2*beta1*epsilon2*(A_base[26])
    p29 = mu2*gamma1*alpha1*omega2*beta1*epsilon1*(A_base[27])
    p30 = mu2*gamma1*alpha1*omega1*beta2*epsilon2*(A_base[28])
    p31 = mu2*gamma1*alpha1*omega1*beta2*epsilon1*(A_base[29])
    p32 = mu2*gamma1*alpha1*omega1*beta1*epsilon2*(A_base[30])
    p33 = mu2*gamma1*alpha1*omega1*beta1*epsilon1*(A_base[31])
    p34 = mu1*gamma2*alpha2*omega2*beta2*epsilon2*(A_base[32])
    p35 = mu1*gamma2*alpha2*omega2*beta2*epsilon1*(A_base[33])
    p36 = mu1*gamma2*alpha2*omega2*beta1*epsilon2*(A_base[34])
    p37 = mu1*gamma2*alpha2*omega2*beta1*epsilon1*(A_base[35])
    p38 = mu1*gamma2*alpha2*omega1*beta2*epsilon2*(A_base[36])
    p39 = mu1*gamma2*alpha2*omega1*beta2*epsilon1*(A_base[37])
    p40 = mu1*gamma2*alpha2*omega1*beta1*epsilon2*(A_base[38])
    p41 = mu1*gamma2*alpha2*omega1*beta1*epsilon1*(A_base[39])
    p42 = mu1*gamma2*alpha1*omega2*beta2*epsilon2*(A_base[40])
    p43 = mu1*gamma2*alpha1*omega2*beta2*epsilon1*(A_base[41])
    p44 = mu1*gamma2*alpha1*omega2*beta1*epsilon2*(A_base[42])
    p45 = mu1*gamma2*alpha1*omega2*beta1*epsilon1*(A_base[43])
    p46 = mu1*gamma2*alpha1*omega1*beta2*epsilon2*(A_base[44])
    p47 = mu1*gamma2*alpha1*omega1*beta2*epsilon1*(A_base[45])
    p48 = mu1*gamma2*alpha1*omega1*beta1*epsilon2*(A_base[46])
    p49 = mu1*gamma2*alpha1*omega1*beta1*epsilon1*(A_base[47])
    p50 = mu1*gamma1*alpha2*omega2*beta2*epsilon2*(A_base[48])
    p51 = mu1*gamma1*alpha2*omega2*beta2*epsilon1*(A_base[49])
    p52 = mu1*gamma1*alpha2*omega2*beta1*epsilon2*(A_base[50])
    p53 = mu1*gamma1*alpha2*omega2*beta1*epsilon1*(A_base[51])
    p54 = mu1*gamma1*alpha2*omega1*beta2*epsilon2*(A_base[52])
    p55 = mu1*gamma1*alpha2*omega1*beta2*epsilon1*(A_base[53])
    p56 = mu1*gamma1*alpha2*omega1*beta1*epsilon2*(A_base[54])
    p57 = mu1*gamma1*alpha2*omega1*beta1*epsilon1*(A_base[55])
    p58 = mu1*gamma1*alpha1*omega2*beta2*epsilon2*(A_base[56])
    p59 = mu1*gamma1*alpha1*omega2*beta2*epsilon1*(A_base[57])
    p60 = mu1*gamma1*alpha1*omega2*beta1*epsilon2*(A_base[58])
    p61 = mu1*gamma1*alpha1*omega2*beta1*epsilon1*(A_base[59])
    p62 = mu1*gamma1*alpha1*omega1*beta2*epsilon2*(A_base[60])
    p63 = mu1*gamma1*alpha1*omega1*beta2*epsilon1*(A_base[61])
    p64 = mu1*gamma1*alpha1*omega1*beta1*epsilon2*(A_base[62])
    p65 = mu1*gamma1*alpha1*omega1*beta1*epsilon1*(A_base[63]) 
    A_h = (p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+\
           p13+p14+p15+p16+p17+p18+p19+p20+p21+\
           p22+p23+p24+p25+p26+p27+p28+p29+p30+\
           p31+p32+p33+p34+p35+p36+p37+p38+p39+\
           p40+p41+p42+p43+p44+p45+p46+p47+p48+\
           p49+p50+p51+p52+p53+p54+p55+p56+p57+\
           p58+p59+p60+p61+p62+p63+p64+p65) 
    
    # Error:
    A_erro = np.array(A_h - L*C)
    error = A_erro.dot(E_)
    
    e1_c = float(error[0])
    e2_c = float(error[1])
    e3_c = float(error[2])
    e4_c = float(error[3])
    e5_c = float(error[4])
    e6_c = float(error[5])
    sol_e = np.array([e1_c,e2_c,e3_c,e4_c,e5_c,e6_c])
    
    return sol_p, sol_c, sol_e, D, memb_func
    
#%% Fourth-order Runge-Kutta integrator:     
# Initial conditions used in the paper for an on-going seizure (Jirsa et al., 2014):
#vec_cond_inic = [0, -5, 3, 0, 0, 0]
#x10 = vec_cond_inic[0]
#y10 = vec_cond_inic[1]
#z0 = vec_cond_inic[2]
#x20 = vec_cond_inic[3]
#y20 = vec_cond_inic[4]
#u0 = vec_cond_inic[5]
    
# Initial conditions for the observer (on-going seizure):
#x10c = x10 - 1
#y10c = y10 - 1
#z0c = z0 - 0.2
#x20c = x20 - 1
#y20c = y20 + 1 
#u0c = u0 - 0.1

# Initial conditions used in the paper for a healthy condition:
x_healthy = np.array([-1.98,-18.66,4.15,-0.88,0,-0.035])
x10 = x_healthy[0]
y10 = x_healthy[1]
z0 = x_healthy[2]
x20 = x_healthy[3]
y20 = x_healthy[4]
u0 = x_healthy[5]

X[:,0] = [x10, y10, z0, x20, y20, u0]

# Initial conditions for the observer (healthy condition):
x10c = x10 - 2
y10c = y10 - 8
z0c = z0 - 0.2
x20c = x20 - 2
y20c = y20 + 0.5
u0c = u0 - 0.4

Xc[:,0] = [x10c, y10c, z0c, x20c, y20c, u0c]

# Initial conditions for the error dynamics:
E10 = x10 - x10c
E20 = y10 - y10c 
E30 = z0 - z0c
E40 = x20 - x20c
E50 = y20 - y20c
E60 = u0 - u0c
E[:,0] = [E10,E20,E30,E40,E50,E60]
    
for k in range(0, N-1):         
    print(str(k))
    k1, kk1, kkk1, D, mf = __dXdt__( X[:,k], Xc[:,k], E[:,k] )
    k2, kk2, kkk2, D, mf = __dXdt__( X[:,k] + k1*(dt/2), Xc[:,k] + kk1*(dt/2), E[:,k] + kkk1*(dt/2) )
    k3, kk3, kkk3, D, mf = __dXdt__( X[:,k] + k2*(dt/2), Xc[:,k] + kk2*(dt/2), E[:,k] + kkk2*(dt/2) )
    k4, kk4, kkk4, D, mf = __dXdt__( X[:,k] + k3*dt, Xc[:,k] + kk3*dt, E[:,k] + kkk3*dt )
    X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) 
    Xc[:,k+1] = Xc[:,k] + (dt/6)*(kk1 + 2*kk2 + 2*kk3 + kk4) 
    E[:,k+1] = E[:,k] + (dt/6)*(kkk1 + 2*kkk2 + 2*kkk3 + kkk4) 
    
    # Nonlinear functions (observer):
    der1[k+1] = D[0]
    der2[k+1] = D[1]
    der3[k+1] = D[2]
    der4[k+1] = D[3]
    der5[k+1] = D[4]
    der6[k+1] = D[5]
    
    # Membership functions:
    mu1_mf[k+1] = mf[0]
    mu2_mf[k+1] = mf[1]
    gamma1_mf[k+1] = mf[2]
    gamma2_mf[k+1] = mf[3]
    alpha1_mf[k+1] = mf[4]
    alpha2_mf[k+1] = mf[5]
    omega1_mf[k+1] = mf[6]
    omega2_mf[k+1] = mf[7]
    beta1_mf[k+1] = mf[8]
    beta2_mf[k+1] = mf[9]
    epsilon1_mf[k+1] = mf[10]
    epsilon2_mf[k+1] = mf[11]
    
#%% Gráficos:    
plt.figure(1)
plt.subplot(231)    
plt.plot(t,X[0,:],'b', linewidth=1.5, label='$Real$ $x_{1}$')
plt.plot(t,Xc[0,:],'r--',linewidth=1.5, label='$Estimated$ $x_{1}$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$x_{1}(t)$', fontsize=15)
plt.ylim(-3, 2)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)       

plt.subplot(232)
plt.plot(t,X[1,:],'b', linewidth=1.5,label='$Real$ $y_{1}$')
plt.plot(t,Xc[1,:],'r--',linewidth=1.5, label='$Estimated$ $y_{1}$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$y_{1}(t)$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(233)
plt.plot(t,X[2,:], 'b', linewidth=1.5, label='$Real$ $z$')
plt.plot(t,Xc[2,:],'r--',linewidth=1.5, label='$Estimated$ $z$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$z(t)$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.ylim(2.5, 4.2)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(234)
plt.plot(t,X[3,:],'b', linewidth=1.5, label='$Real$ $x_{2}$')
plt.plot(t,Xc[3,:],'r--',linewidth=1.5, label='$Estimated$ $x_{2}$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$x_{2}(t)$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.ylim(-1.5,1)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)   

plt.subplot(235)
plt.plot(t,X[4,:],'b', linewidth=1.5, label='$Real$ $y_{2}$')
plt.plot(t,Xc[4,:],'r--',linewidth=1.5, label='$Estimated$ $y_{2}$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$y_{2}(t)$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(236)
plt.plot(t,X[5,:],'b', linewidth=1.5, label='$Real$ $z$')
plt.plot(t,Xc[5,:],'r--',linewidth=1.5, label='$Estimated$ $u$')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$u(t)$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()  
plt.tick_params(axis='both', which='major', labelsize=15)
    
#%% Error:
plt.figure(2)
plt.subplot(231)
plt.plot(t, E[0,:],'b',linewidth=2)     
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{x_{1}}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()  
plt.tick_params(axis='both', which='major', labelsize=15)
        
plt.subplot(232)
plt.plot(t, E[1,:],'b',linewidth=2)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{y_{1}}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(233)
plt.plot(t, E[2,:],'b',linewidth=2)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{z}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(234)
plt.plot(t, E[3,:],'b',linewidth=2)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{x_{2}}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(235)
plt.plot(t, E[4,:],'b',linewidth=2)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{y_{2}}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(236)
plt.plot(t, E[5,:],'b',linewidth=2)
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$E_{u}$', fontsize=15)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Norm of the error:
EE = np.sqrt( E[0,:]**2 + E[1,:]**2 + E[2,:]**2 + E[3,:]**2 + E[4,:]**2 + E[5,:]**2 )

plt.figure(3)
plt.plot(t, EE , 'b', linewidth=2)
plt.grid()
plt.xlim(t[0], t[-1])
plt.ylim(0, 6)
plt.xlabel('$t$ $[s]$',fontsize=25)
plt.ylabel('$\parallel \mathbf{e}(t) \parallel$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Weighting functions:
plt.figure(4)
plt.subplot(321)
plt.plot(t, mu1_mf, 'b', label='$\mu_{1}$')
plt.plot(t, mu2_mf, 'r--', label='$\mu_{2}$')
plt.ylabel('$MF_{1}^{e}$', fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(322)
plt.plot(t, gamma1_mf, 'b', label='$\gamma_{1}$')
plt.plot(t, gamma2_mf, 'r--', label='$\gamma_{2}$')
plt.ylabel('$MF_{2}^{e}$', fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(323)
plt.plot(t, alpha1_mf, 'b', label='$\alpha_{1}$')
plt.plot(t, alpha2_mf, 'r--', label='$\alpha_{2}$')
plt.ylabel('$MF_{3}^{e}$', fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(324)
plt.plot(t, omega1_mf, 'b', label='$\omega_{1}$')
plt.plot(t, omega2_mf, 'r--', label='$\omega_{2}$')
plt.ylabel('$MF_{4}^{e}$', fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(325)
plt.plot(t, beta1_mf, 'b', label='$\beta_{1}$')
plt.plot(t, beta2_mf, 'r--', label='$\beta_{2}$')
plt.ylabel('$MF_{5}^{e}$', fontsize=25)
plt.xlabel('$t$ $[s]$',fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(326)
plt.plot(t, epsilon1_mf, 'b', label='$\epsilon_{1}$')
plt.plot(t, epsilon2_mf, 'r--', label='$\epsilon_{2}$')
plt.ylabel('$MF_{6}^{e}$', fontsize=25)
plt.xlabel('$t$ $[s]$',fontsize=25)
plt.ylim(0,1)
plt.xlim(t[0], t[-1])
plt.grid()
plt.show()
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Reconstructed nonlinear functions:
plt.figure(5)
plt.subplot(321)
plt.plot(t, der1, 'b')
plt.plot(t, np.array(mu1_mf)*zeta1_min + np.array(mu2_mf)*zeta1_max, 'r--')
plt.ylabel('$f_{1}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-30,5)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(322)
plt.plot(t, der2, 'b')
plt.plot(t, np.array(gamma1_mf)*zeta2_min + np.array(gamma2_mf)*zeta2_max, 'r--')
plt.ylabel('$f_{2}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-20,30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(323)
plt.plot(t, der3, 'b')
plt.plot(t, np.array(alpha1_mf)*zeta3_min + np.array(alpha2_mf)*zeta3_max, 'r--')
plt.ylabel('$f_{3}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-3.5,-0.5)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(324)
plt.plot(t, der4, 'b')
plt.plot(t, np.array(omega1_mf)*zeta4_min + np.array(omega2_mf)*zeta4_max, 'r--')
plt.ylabel('$f_{4}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-2,3)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(325)
plt.plot(t, der5, 'b')
plt.plot(t, np.array(beta1_mf)*zeta5_min + np.array(beta2_mf)*zeta5_max, 'r--')
plt.xlabel('$t$ $[s]$', fontsize=25)
plt.ylabel('$f_{5}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-5,2)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(326)
plt.plot(t, der6, 'b')
plt.plot(t, np.array(epsilon1_mf)*zeta6_min + np.array(epsilon2_mf)*zeta6_max, 'r--')
plt.xlabel('$t$ $[s]$', fontsize=25)
plt.ylabel('$f_{6}^{e}$', fontsize=25)
plt.xlim(t[0], t[-1])
plt.ylim(-0.1,0.7)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

#%% Bursters:
fig = plt.figure(6)
ax = plt.axes(projection='3d')

xline = Xc[2]
yline = Xc[3]
zline = Xc[4]
ax.plot3D(xline, yline, zline, 'r', linewidth=2)

ax = plt.gca()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

ax.set_xlabel('$\hat{z}$', fontsize=30)
ax.set_ylabel('$\hat{x}_{2}$', fontsize=30)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
ax.set_zlabel('$\hat{y}_{2}$', fontsize=30, rotation = 0)