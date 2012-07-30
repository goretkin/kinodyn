# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:05:03 2012

@author: gustavo
"""

import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
from lqr_tools import AQR,simulate_lti_fb, lqr_dim

def ctfh_lqr(A,B,Q,R,Qf,ts):
    (n,m) = lqr_dim(A,B,Q,R)
    assert Qf.shape == (n,n)

    def matrix_to_state(m):
        return np.array(m).flatten()
    
    def state_to_matrix(s):
        return np.matrix(np.reshape(s,newshape=(n,n)))
        
    def dSdt(S_,t):
        S = state_to_matrix(S_)
        dSdt_ = -S * A - A.T * S + S * B * R.I * B.T * S - Q
        return matrix_to_state(dSdt_)
        
    def back_dSdt(S,t):
        return -dSdt(S,-t)
        
    S_ts = scipy.integrate.odeint(func=back_dSdt,y0=matrix_to_state(Qf),t=ts)
    Ss = np.zeros(shape=(len(ts),n,n))
    for i in range(len(ts)):
        Ss[i] = state_to_matrix(S_ts[len(ts)-i-1])
    return Ss
    
def ctfh_lqr_dual(A,B,Q,R,Qf_inv,ts):
    (n,m) = lqr_dim(A,B,Q,R)
    assert Qf_inv.shape == (n,n)
    
    def matrix_to_state(m):
        return np.array(m).flatten()
    
    def state_to_matrix(s):
        return np.matrix(np.reshape(s,newshape=(n,n)))
        
    def dSdt(S_,t):
        S = state_to_matrix(S_)
        dSdt_ = A * S + S * A.T - B * R.I * B.T  + S * Q * S
        return matrix_to_state(dSdt_)
        
    def back_dSdt(S,t):
        return -dSdt(S,-t)
        
    S_ts = scipy.integrate.odeint(func=back_dSdt,y0=matrix_to_state(Qf_inv),t=ts)
    Ss = np.zeros(shape=(len(ts),n,n))
    for i in range(len(ts)):
        Ss[i] = state_to_matrix(S_ts[len(ts)-i-1])
    return Ss    


if __name__=='__main__' :
    #continuous-time dynamics. 
    # x = [vel,pos].T
    
    d= -0e-1    #damping

    k  = 0      #spring
    A = np.matrix([[d,-k],
                     [1,0]])
    
    B = np.matrix([[1],
                   [0]])
                   
    Q = np.matrix([[1,0],
                   [0,1]])

    R = np.matrix([[1]]) *1e2
    
    
    T = 5 #time horizon for LQR
    ts = np.linspace(0,T,8000)
    

    #final-value constrained LQR
    Qdualfinal = np.zeros_like(Q)
    cost_to_go_dual = ctfh_lqr_dual(A,B,Q,R,Qdualfinal,ts)
   
    cost_to_go_dual_inv = np.empty_like(cost_to_go_dual)

    for i in range(cost_to_go_dual.shape[0]):
        print np.linalg.cond(cost_to_go_dual[i])
        cost_to_go_dual_inv[i] = np.linalg.pinv(cost_to_go_dual[i])
    

    #drive LTI CT finite-horizon LQR
    n_ts = cost_to_go_dual_inv.shape[0]
    n,m = lqr_dim(A,B,Q,R)
    gain_schedule = np.zeros(shape=(n_ts,m,n))
    for i in range(n_ts):
         gain_schedule[i] = -np.dot((R.I * B.T), cost_to_go_dual_inv[i])
    
    ts_sim = np.linspace(0,T,1000)
    x0 = np.array([1,0])
    desired = np.array([2,-2])

    traj =simulate_lti_fb(A,B,x0=x0,ts=ts_sim,
                          gain_schedule=gain_schedule,
                          gain_schedule_ts=ts,
                          setpoint = desired)    
    
    print traj[-1]
    plt.plot(ts_sim,traj[:,0])
    plt.plot(ts_sim,traj[:,1])


if __name__=='__main__' and False:
    #continuous-time affine dynamics. 
    # x = [vel,pos].T
    
    d= -0e-1    #damping
    g = 0      #gravity
    k  = 0      #spring
    A = np.matrix([[d,-k],
                     [1,0]])
    
    B = np.matrix([[1],
                   [0]])

    c = np.matrix([[g],
                   [0]])
                   
    Q = np.matrix([[1,0],
               [0,1]])

    R = np.eye(1)*1e2
    
    desired = np.matrix([[0],
                     [1]])   

    q = -np.dot(Q,desired)
    
    r=np.matrix([0]).T
    
    (Ah,Bh,Qh,Rh,u_offset) =  AQR(A,B,c,Q,q,R,r,ctdt='ct')
    
    T = 20 #time horizon for LQR
    ts = np.linspace(0,T,4000)
    cost_to_go = ctfh_lqr(Ah,Bh,Qh,Rh,Qh,ts)
    
    #cost to go is supposed to be symmetric, but this isn't constrained
    #by the integration
    
    total_assym_error = np.sum(np.abs(cost_to_go-np.transpose(cost_to_go,axes=[0,2,1])))
    print 'asymmetry error in cost to go is: ',total_assym_error
    
    cost_to_go_dual = ctfh_lqr_dual(Ah,Bh,Qh,Rh,Qh.I,ts)
   
    cost_to_go_dual_inv = np.empty_like(cost_to_go_dual)
    for i in range(cost_to_go_dual.shape[0]):
        cost_to_go_dual_inv[i] = np.linalg.inv(cost_to_go_dual[i])
    
    #should compute relative error
    total_error = np.sum(np.abs(cost_to_go_dual_inv-cost_to_go))
    print 'difference between dual and primal is: ', total_error
    print 'difference between dual and primal per value is: ', total_error/len(ts)/Ah.shape[0]**2
    
    """when comparing the dual and primal cost-to-go matrices, 
    it might not be important what the value of the lower-right entry is, since
    this corresponds to an constant-offset in
    """
    
    #drive LTI CT finite-horizon LQR
    
    
    n_ts = cost_to_go.shape[0]
    gain_schedule = np.zeros(shape=(n_ts,Bh.shape[1],Ah.shape[0]))
    for i in range(n_ts):
         gain_schedule[i] = -np.dot((Rh.I * Bh.T), cost_to_go[i])
    
    ts_sim = np.linspace(0,T,1000)    
    x0 = np.array([0,0,1])
    traj =simulate_lti_fb(Ah,Bh,x0=x0,ts=ts_sim,
                          gain_schedule=gain_schedule,
                          gain_schedule_ts=ts)    
    
    print traj[-1]
    plt.plot(ts_sim,traj[:,0])
    plt.plot(ts_sim,traj[:,1])
    
    #final-value constrained LQR
    cost_to_go_dual = ctfh_lqr_dual(Ah,Bh,Qh,Rh,np.zeros_like(Qh),ts)
    
    cost_to_go_pinv = np.empty_like(cost_to_go_dual)
    for i in range(cost_to_go_dual.shape[0]):
        cost_to_go_pinv[i] = np.linalg.pinv(cost_to_go_dual[i])
        


