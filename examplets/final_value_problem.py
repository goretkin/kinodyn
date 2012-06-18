# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:05:03 2012

@author: gustavo

typically one solves xdot = f(x) by choosing some initial x0 and integrating forward.
here we see what can happen if the problem is instead we're given xf at some final time and we
integrate backwards.

with smarter integrators, you'd probably get better results
"""

import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import scipy.optimize

eps = 1.0
def dSdt(S,t):
    """
    the dynamics of the system, as expected by scipy.integrate.odeint
    this is the Van der Pol oscillator http://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    chosen because it's non-conservative, and probably impossible to stably time-reverse?
    """
    x,xdot = S
    xddot = eps * (1-x**2)*xdot - x
    return np.array([xdot,xddot])
    
def back_dSdt(S,t):
    """
    the time reverse system of dSdt
    """
    return -dSdt(S,-t)
    

Tf = 22
ts = np.linspace(0,Tf,5000)

BEHAVIOR = 'BLOW UP'
#BEHAVIOR = 'DECAY'

if BEHAVIOR == 'DECAY':
    #some points that start at different parts of the limit cycle, approximately
    #for length-30 integration, all of these have time-reverses that decay.
    state0 = [2,0]
    state0 = [-1,1]  
    state0 = [1,-1]

    #start inside the limit cycle -- takes a while to pump up, but time reverse still decays faster
    state0 = [.1,.1] 

if BEHAVIOR == 'BLOW UP':
    #use a shooting method to find an initial state that minimizes the value of the y[1] (the value of the second dimension) at the final time.
    def objective(x):
        y = (scipy.integrate.odeint(func=dSdt,y0=x,t=ts))[-1]
        return y[1]

    state0 = scipy.optimize.fmin(objective,[2,0])




#integrate the forward system
y = scipy.integrate.odeint(func=dSdt,y0=state0,t=ts)

#integrate backwards, starting from the final state found by forward integration
back_y = scipy.integrate.odeint(func=back_dSdt,y0=y[-1],t=ts)

#in the perfect world, y and back_y agree (except that back_y is reversed, so really back_y[::-1] should match y. 

fig_ts = plt.figure(None)

ax_ts1 = fig_ts.add_subplot(2,1,1)
ax_ts2 = fig_ts.add_subplot(2,1,2,sharex=ax_ts1)

ax_ts1.plot(y[:,0],'b')
ax_ts1.plot(back_y[::-1,0],'g')
ax_ts1.set_ylim(-5,5)

ax_ts2.plot(y[:,1])
ax_ts2.plot(back_y[::-1,1])
ax_ts2.set_ylim(-5,5)

fig_phase = plt.figure(None)
ax_phase =  fig_phase.add_subplot(1,1,1)
ax_phase.plot(y[:,0],y[:,1],'b')
ax_phase.plot(back_y[::-1,0],back_y[::-1,1],'g')

ax_phase.set_xlim(-5,5)
ax_phase.set_ylim(-5,5)
plt.show()
