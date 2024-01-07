# -*- coding: utf-8 -*-
"""
@author: João Angelo Ferres Brogin

São Paulo State University (UNESP) - School of Engineering of Ilha Solteira (FEIS)
Group of Intelligent Materials and Systems (GMSINT)
Post-Graduation Program in Mechanical Engineering (PPGEM)

The following code was implemented with the purpose of designing and applying a fuzzy observer to estimate the unmeasurable variables
of the dynamic model known as Epileptor (Jirsa et al., 2014), which represents the main features of epileptiform activity. 
In this file, the Linear Matrix Inequalities (Tanaka & Wang, 2004) are defined as an optimization problem to obtain the gains of the controller for the spring-mass-damper system.
Make sure to install the following packages before running it, please: CVXOPT and PICOS (Sagnol, 2012).

The main results obtained can be found in the article entitled (Appendix D): 
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
m = 1
K = 1
c = 0

#%% Linear system:
A = np.array([  [      0,     1    ], \
                [   -K/m,   -c/m   ] ]) 

B = np.array([ 0, \
               1/(m)])  

N = len(A)

#%% Input matrices:
alpha = 0.3
beta = 0
I = alpha  *  np.identity(N)
H = beta *  np.ones((2,2))

#%% Optimization problem:
prob = pic.Problem()

BB = pic.new_param('B', cvx.matrix(B))
AA = pic.new_param('A', cvx.matrix(A) )    
HH = pic.new_param('H', cvx.matrix(H) )
II = pic.new_param('I', cvx.matrix(I) )

# Variables to be found:
Gx = prob.add_variable('Gx', (1,N))
X = prob.add_variable('X', (N,N), vtype='symmetric')

# Linear Matrix Inequalities as constraints:
prob.add_constraint(X *  (AA.T + II.T + HH.T ) - Gx.T * BB.T + ( AA + II + HH ) * X - BB * Gx << 0 )
prob.add_constraint(X >> 0)

# Solver:
prob.solve(verbose=1)
print('Status: ' + prob.status)

X = np.matrix(X.value)
Gx = np.matrix(Gx.value)

P = np.matrix(X).I
G = np.matrix(Gx).dot(P)