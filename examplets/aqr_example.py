# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:42:53 2012

@author: gustavo
"""

from __future__ import division

import examplets

import control
import yottalab
import scipy
import scipy.signal
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

from ct_fh_lqr import ctfh_lqr, ctfh_lqr_dual
from lqr_tools import AQR, LQR_QP, simulate_lti_fb_dt, simulate_lti_fb    
from lqr_tools import dtfh_lqr, dtfh_lqr_dual

#continuous-time affine dynamics. 
d= -0e-1    #damping
g = 0      #gravity
k  = 0      #spring
Act = np.matrix([[d,-k,g],
                 [1,0,0],
                 [0,0,0]])


Bct = np.matrix([[1],
               [0],
               [0]])

# x = [vel,pos].T

Ts = 1

a = Act
b = Bct
nb=1
n=3
m=1

ztmp=np.zeros((nb,n+nb))
tmp=np.hstack((a,b))
tmp=np.vstack((tmp,ztmp))
tmp=scipy.linalg.expm(tmp*Ts)
A_affine=tmp[0:n,0:n]
B_affine=tmp[0:n,n:n+nb]

A = A_affine[0:2,0:2]
B = B_affine[0:2]
c = np.matrix(A_affine[0:2,2]).T

Q = np.matrix([[1,0],
               [0,1]])
               
R = np.eye(1)*1e2


desired = np.matrix([[0],
                     [1]])                     

q = -np.dot(Q,desired)

r=np.matrix([0]).T

(Ah,Bh,Qh,Rh,pk) = AQR(A=A,B=B,c=c,Q=Q,R=R,q=q,r=r,ctdt='dt')

T = 50
x0 = np.array([0e0,0])

DUAL = False
if not DUAL:
    gain_matrices,cost_to_go_matrices = dtfh_lqr(Ah,Bh,Qh,Rh,N=T-1,
                                               Q_terminal=Qh*1e6)
else:
    gain_matrices,cost_to_go_matrices = dtfh_lqr_dual(Ah,Bh,Qh,Rh,N=T-1,
                                        Q_terminal_inv=Qh*0e0)


dp_xs,dp_us = simulate_lti_fb_dt(A=Ah,B=Bh,x0=np.concatenate([x0,[1]]),
                   gain_schedule=-gain_matrices,T=T)

qp_solution,(QP_P,QP_q,QP_A,QP_B),qp_xs,qp_us = LQR_QP(Ah,Bh,Qh,Rh,T=T,x0=np.concatenate([x0,[1]]),                     
                     #xT=np.concatenate([[10,0],[1]])
                     #xT=np.concatenate([[1,0],[1]])
                     xT=np.concatenate([desired.flat,[1]])
                     )

#qp_solution,(QP_P,QP_q,QP_A,QP_B) = LQR_QP(Ah,Bh,Qh,Rh,T=T,x0=np.concatenate([x0,[1]]))

#dsol_qp = np.array(qp_solution['x']).reshape(-1,n+m).T
#nextX = np.dot(A,dsol_qp[0:2,:]) + np.dot(B,dsol_qp[3:,:])+c
#dynamics_constraint_error = nextX[0:2,0:-1] - dsol_qp[0:2,1:]

########
#np.sum(np.abs((np.matrix(QP_A) * qp_vars)-QP_B))
#dp_vars.T*np.matrix(QP_P)*dp_vars


def cost_traj(x,u):
    c = 0
    for i in range(x.shape[1]):
        xR = np.matrix(x[:,i]).T
        xL = np.matrix(x[:,i])
        c += xL*Qh*xR
        
    for i in range(u.shape[1]):
        uR = np.matrix(u[:,i]).T
        uL = np.matrix(u[:,i])
        c += uL*Rh*uR
        
    return c

qp_cost = cost_traj(qp_xs,qp_us)
dp_cost = cost_traj(dp_xs,dp_us)

qp_vars = np.array(qp_solution['x'])
#transforms the DP solution into the vector form the QP uses.
#dsol_dp = np.concatenate([dp_solution[0],dp_solution[1]],axis=0)
#dp_vars = dsol_dp.reshape(-1,1,order='F')

#print 'qp_cost',qp_cost,'qp_cost',qp_solution['dual objective']*2

ct_gain_samples = 5000

dsol = np.matrix(np.zeros(shape=(2,T)))
dcontrol = np.matrix(np.zeros(shape=(1,T)))
dsol[:,0] = np.matrix(x0).T
dsol1 = np.vstack( (dsol,np.ones(shape=(1,T))) )

cost_to_go = np.zeros(shape=(ct_gain_samples))
       

DUAL = True
ts = np.linspace(0,T*Ts,ct_gain_samples)

if DUAL:
    cost_to_go_matrices_ct = ctfh_lqr_dual(Act,Bct,Qh,Rh,
                                           np.zeros_like(Qh.I),
                                            ts)
else:
    cost_to_go_matrices_ct = ctfh_lqr(Act,Bct,Qh,Rh,Qh,ts)

gain_schedule_ct = np.zeros(shape=(ct_gain_samples,m,n))    


for k in range(0,ct_gain_samples):
    if DUAL:
        ctg = np.linalg.pinv(cost_to_go_matrices_ct[k])
    else:
        ctg = cost_to_go_matrices_ct[k]
        
    #a = np.dot(ctg,dsol1[:,k])
    #cost_to_go[k] = np.dot(dsol1[:,k].T,a)
    gain_schedule_ct[k] = -Rh.I * Bct.T * np.matrix(ctg)

traj = simulate_lti_fb(A=Act,B=Bct,x0=np.concatenate([x0,[1]]),
                ts=ts,gain_schedule=gain_schedule_ct,
                gain_schedule_ts=ts)

plt.figure(None)
plt.subplot(3,1,1)
#plt.plot(dsol[0,:].T,'g-') #velocity
#plt.plot(dsol[1,:].T,'b-') #position
#plt.plot(dcontrol[0,:].T,'r-') #imparted acceleration
plt.title('DP')
plt.plot(dp_xs[0,:].T,'g-') #velocity
plt.plot(dp_xs[1,:].T,'b-') #position
plt.plot(dp_us[0,:].T,'r-') #imparted acceleration
print dp_xs
plt.axhline(color='k')


plt.subplot(3,1,2)
plt.title('QP')
plt.plot(qp_xs[0,:].T,'g-') #velocity
plt.plot(qp_xs[1,:].T,'b-') #position
plt.plot(qp_us[0,:].T,'r-') #imparted acceleration
plt.axhline(color='k')

plt.subplot(3,1,3)
#plt.plot(cost_to_go,'g-')
plt.plot(traj.T[0,:],'g-')
plt.plot(traj.T[1,:],'b-')

plt.axhline(color='k')
