"""
How finite-time LQR behaves with an uncontrollable system
"""
import examplets
import control
import yottalab
import scipy
import scipy.signal

from lqr_tools import dtfh_lqr, simulate_lti_fb_dt

import numpy as np
import matplotlib.pyplot as plt

A = np.matrix([[2,0],
              [0,2]])

B = np.matrix([1,-1]).T


Q = np.matrix([[1,0],
               [0,1]])
R = np.eye(1)

T = 20
gain_schedule,cost_to_go_schedule = dtfh_lqr(A,B,Q,R,T-1)

x0 = np.array([1,1])

xsol, usol = simulate_lti_fb_dt(A,B,x0,-gain_schedule,T)

plt.figure(None)
plt.plot(xsol[0],xsol[1])
plt.plot(xsol[0],xsol[1],'o')

fig = plt.figure(None)
fig.add_subplot(3,1,1).plot(xsol[0])
fig.add_subplot(3,1,2).plot(xsol[1])
fig.add_subplot(3,1,3).plot(usol[0,:])








