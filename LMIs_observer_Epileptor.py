# -*- coding: utf-8 -*-
"""
@author: João Angelo Ferres Brogin

São Paulo State University (UNESP) - School of Engineering of Ilha Solteira (FEIS)
Group of Intelligent Materials and Systems (GMSINT)
Post-Graduation Program in Mechanical Engineering (PPGEM)

The following code was implemented with the purpose of designing and applying a fuzzy observer to estimate the unmeasurable variables
of the dynamic model known as Epileptor (Jirsa et al., 2014), which represents the main features of epileptiform activity. 
In this file, the Linear Matrix Inequalities (Tanaka & Wang, 2004) are defined as an optimization problem to obtain the gains of the fuzzy observer. 
Make sure to install the following packages before running it, please: CVXOPT and PICOS (Sagnol, 2012).

The main results obtained can be found in the article entitled: 
Brogin, J. A. F., Faber, J. and Bueno, D. D. "Burster Reconstruction Considering Unmeasurable Variables in the Epileptor Model".

Plase cite this code as: 
Brogin, J. A. F., Faber, J. and Bueno, D. D. (2021) Burster reconstruction considering unmeasurable variables in the Epileptor model [Computer code]. 

References:
[1] Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. Brain, 137(8), 2210-2230.
[2] Tanaka, K., & Wang, H. O. (2004). Fuzzy control systems design and analysis: a linear matrix inequality approach. John Wiley & Sons.
[3] Sagnol, G. PICOS: A Python Interface for Conic Optimization Solvers. Available online: http://picos.zib.de (accessed on 31 March 2021)

"""

import numpy as np
import picos as pic
import cvxopt as cvx

#%% Parameters of the model:
tau1 = 1
tau2 = 10   
tau0 = 2857.0
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Maximum and minimum values of the nonlinear functions:
af = 0.1 # for a more conservative approach, increase this value

zeta1_max = 2 + af 
zeta1_min = -24.1 - af
zeta1 = [0,zeta1_min,zeta1_max]

zeta2_max = 21 + af
zeta2_min = -16 - af 
zeta2 = [0,zeta2_min,zeta2_max]

zeta3_max = -0.9 + af 
zeta3_min = -3.1 - af 
zeta3 = [0,zeta3_min,zeta3_max]

zeta4_max = 2.1 + af 
zeta4_min = -1.6 - af 
zeta4 = [0,zeta4_min,zeta4_max]

zeta5_max = 1 + af 
zeta5_min = -3.8 - af 
zeta5 = [0,zeta5_min,zeta5_max]

zeta6_max = 0.6 + af 
zeta6_min = 0 - af
zeta6 = [0,zeta6_min,zeta6_max]

#%% Submodels:
Nnl = 6
n_models = 2**Nnl
A_base = [None]*n_models

A_base[0] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ]) 
A_base[1] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])
A_base[2] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ]) 
A_base[3] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ]) 
A_base[4] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ]) 
A_base[5] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[6] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[7] = np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[8] = np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[9] = np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[10]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[11]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[12]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[13]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[14]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[15]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[16]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[17]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[18]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[19]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[20]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[21]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[22]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[23]= np.array([[         zeta1[2],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[24]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[25]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[26]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[27]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[28]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[29]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[30]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[31]= np.array([[         zeta1[2],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[32]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[33]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[34]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[35]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[36]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[37]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[38]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[39]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[40]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[41]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[42]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[43]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[44]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[45]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[46]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[47]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[2],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[48]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[49]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[50]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[51]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[52]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[53]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[54]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[55]= np.array([[         zeta1[1],         1,    zeta3[2],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[56]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[57]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[58]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[59]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[2],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[60]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[61]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[2],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[62]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[2],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
A_base[63]= np.array([[         zeta1[1],         1,    zeta3[1],      zeta4[1],            0,            0], \
                      [         zeta2[1],   -1/tau1,           0,             0,            0,            0], \
                      [           4/tau0,         0,     -1/tau0,             0,            0,            0], \
                      [                0,         0,        -0.3,      zeta5[1],           -1,            2], \
                      [                0,         0,           0,      zeta6[1],      -1/tau2,            0], \
                      [        0.1*gamma,         0,           0,             0,            0,       -gamma] ])  
    
#%% Parameters for the decay rate:
# Epileptiform and healthy activities (transition):
alpha = 0.000002*2
beta =  0.03
r = 0.5

# On-going seizures:
#alpha = 0.000001*2
#beta = 0.2
#r = 2

#%% Optimization problem:
prob = pic.Problem()
                        
A_base_cvx = []     
CC = np.zeros((1,6))
CC[0][0] = -1
CC[0][3] = 1
I = np.identity(6)              
H = np.ones((6,6))

C = pic.new_param('C', cvx.matrix(CC))
II = pic.new_param('I', cvx.matrix(I))
HH = pic.new_param('H', cvx.matrix(H))

for op in range(0, n_models):
    Am = A_base[op]
    A_base_cvx.append( pic.new_param('A'+str(op+1), cvx.matrix(Am) ) )

M = prob.add_variable('M', (6,1) )
P = prob.add_variable('P', (6,6), vtype='symmetric')

# Restrictions (LMIs):
for qq in range(0, n_models):    
    prob.add_constraint( ( A_base_cvx[qq] + beta*HH + alpha*II ).T * P + P * ( A_base_cvx[qq] + beta*HH + alpha*II ) - M * C - C.T * M.T << 0  )
    prob.add_constraint( ( (- r * P & P * A_base_cvx[qq] - M * C )//( A_base_cvx[qq].T * P - C.T * M.T & -r * P) ) << 0 )
    
prob.add_constraint( P >> 0 )

#%% Solver:
prob.solve(verbose = 1)
print('Status: ' + prob.status)

#%% Variables:
P = np.matrix(P.value)
M = np.matrix(M.value)
C = np.matrix(C.value)
L = P.I * M

P2 = P