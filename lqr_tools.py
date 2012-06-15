# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:33:30 2012

@author: gustavo
"""
import numpy as np

def dtfh_lqr(A,B,Q,R,N,Q_terminal=None):
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(Q)
    R = np.matrix(R)
    for x in [A,B,Q,R]:
        assert len(x.shape)==2
    n=A.shape[0]
    assert n==A.shape[1]
    m=B.shape[1]
    assert B.shape[0]==n
    assert Q.shape[0]==n
    assert Q.shape[1]==n
    assert R.shape[0]==m
    assert R.shape[1]==m
    
    if(Q_terminal is None):
        Q_terminal = Q
    assert Q_terminal.shape[0]==Q_terminal.shape[1]==n
    Ps = np.zeros(shape=(N,n,n))

    Fs = np.zeros(shape=(N,m,n)) #gain matrices
    
    Ps[N-1]=Q_terminal     #terminal cost
    pinv = np.linalg.pinv
    #the iteration starts at k and updates the value for k-1
    #as such, the iteration ranges from N-1 to 1 (filling in the values N-2 to 0)
    #N-1 is initialized.
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k])
        #Ps[k-1]=Q + A.T * (Pk - Pk*B*(R+B.T*Pk*B).I * B.T*Pk)*A
        Ps[k-1]=Q + A.T * (Pk - Pk*B* pinv(R+B.T*Pk*B) * B.T*Pk)*A
        
    #there is no propogation here, this is just a non-vectorized way
    #to calculate the gain matrices. As such, the iteration ranges from
    #N-1 to 0
    for k in range(N-1,-1,-1):
        Pk = np.matrix(Ps[k])      
        #Fs[k] = (R+B.T*Pk*B).I * B.T * Pk * A
        Fs[k] = pinv(R+B.T*Pk*B) * B.T * Pk * A

    return (Fs,Ps)



def _ftdlqr(A,B,Q,R,N,Q_terminal=None):
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(Q)
    R = np.matrix(R)
    for x in [A,B,Q,R]:
        assert len(x.shape)==2
    n=A.shape[0]
    assert n==A.shape[1]
    m=B.shape[1]
    assert B.shape[0]==n
    assert Q.shape[0]==n
    assert Q.shape[1]==n
    assert R.shape[0]==m
    assert R.shape[1]==m
    
    if(Q_terminal is None):
        Q_terminal = Q
    assert Q_terminal.shape[0]==Q_terminal.shape[1]==n
    Ps = np.zeros(shape=(N,n,n)) #cost-to-go
    Fs = np.zeros(shape=(N,m,n)) #gain matrices
    
    Ps[N-1]=Q_terminal     #terminal cost
        
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k])
        Fkz = (R+B.T*Pk*B).I * B.T * Pk * A #F_{k minus 1}
        Ps[k-1]= Q + Fkz.T*R*Fkz + ((A-B*Fkz).T)*Pk*(A-B*Fkz)
        Fs[k-1] = Fkz
        
        #form on wikipedia
        Ps[k-1] = Q + A.T * (Pk -Pk*B*((R+B.T*Pk*B).I) *B.T*Pk)*A
        

    return (Fs,Ps)

def dtfh_lqr_dual(A,B,Q,R,N,Q_terminal_inv=None):
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(Q)
    R = np.matrix(R)
    for x in [A,B,Q,R]:
        assert len(x.shape)==2
    n=A.shape[0]
    assert n==A.shape[1]
    m=B.shape[1]
    assert B.shape[0]==n
    assert Q.shape[0]==n
    assert Q.shape[1]==n
    assert R.shape[0]==m
    assert R.shape[1]==m
    
    if(Q_terminal_inv is None):
        Q_terminal_inv = Q.I
    assert Q_terminal_inv.shape[0]==Q_terminal_inv.shape[1]==n
    Ps = np.zeros(shape=(N,n,n)) #inverse cost-to-go
    Fs = np.zeros(shape=(N,m,n)) #gain matrices
    
    Ps[N-1]=Q_terminal_inv     #terminal cost
        
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k])
        print k,Pk
        Ps[k-1] = (Q + A.T * (Pk.I -Pk.I*B*((R+B.T*Pk.I*B).I) *B.T*Pk.I)*A).I
        
    
    for k in range(N-1,-1,-1):
        Pk = np.matrix(Ps[k]).I      
        Fs[k] = (R+B.T*Pk*B).I * B.T * Pk * A
        
    return (Fs,Ps)


def AQR(A,B,c,Q,q,R,r,ctdt='dt'):
    """
    given the following LQR problem
    xdot = Ax + Bu + c  or x_k+1 = Ax_k + Bu_k + c
    with cost functional x^T Q x + 2 q^T x + u^T R u + 2r^T u
    
    return matrices Ah,Bh,Qh,Rh, specifies the regular LQR problem
    in the augmented state space xh = [x^T 1]^T
    The control law is accordingly uh = M * xh 
    u = uh - R^(-1) r
    """
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(Q)
    R = np.matrix(R)
    for x in [A,B,Q,R]:
        assert len(x.shape)==2
        
    c = np.matrix(c)
    q = np.matrix(q)
    r = np.matrix(r)
    
    n=A.shape[0]    #state dimension
    assert n==A.shape[1]
    m=B.shape[1]    #control dimension
    assert n == B.shape[0] == Q.shape[0] == Q.shape[1]
    assert m == R.shape[0] == R.shape[1] == r.shape[0]
    assert n == c.shape[0] == q.shape[0]
    
    nh = n + 1
    
    Ah = np.zeros(shape=(nh,nh))
    Bh = np.zeros(shape=(nh,m))
    Qh = np.zeros(shape=(nh,nh))

    
    Ah[0:n,0:n] = A
    test = c - B*R.I*r
    Ah[0:n,n] = test[:].T
    
    if(ctdt=='dt'):
        Ah[n,n]=1
    
    Bh[0:n,:] = B
    
    Qh[0:n,0:n]=Q
    Qh[n,0:n] = q.T
    Qh[0:n,n] = q.T
    Qh[n,n] = 10 #arbitrary
    
    Rh = R
    
    Ah = np.matrix(Ah)
    Bh = np.matrix(Bh)
    Qh = np.matrix(Qh)
    Rh = np.matrix(Rh)
    
    return (Ah,Bh,Qh,Rh,-R.I*r)

import scipy.integrate
import scipy.interpolate 

def closed_loop_dynamics(dynamics,feedback):
    #dynamics is dxdt = f(x,u,t)
    #feedback is u = g(x,t)
    
    def dxdt(x,t):
        u = feedback(x,t)
        return np.array(dynamics(x,u,t))
        
    return dxdt
     
def simulate_lti_fb(A,B,x0,ts,gain_schedule,gain_schedule_ts=None):
    """
    gain_schdule is either one gain matrix or a sequence of gain matrices
    
    if a schedule:
    gain_schedule.shape = (T,m,n) array, where T is the number of 
    points in the time the gain is defined for and n is the dimension of 
    the state space
    
    """
    A = np.array(A)
    B = np.array(B)
    
    n = A.shape[0]
    m = B.shape[1]
    assert n == A.shape[1] == B.shape[0] 
    if gain_schedule_ts is not None:
        assert m == gain_schedule.shape[1]
        assert n == gain_schedule.shape[2]
        T = gain_schedule_ts.shape[0]
        assert T == gain_schedule.shape[0]
    else:
        assert m == gain_schedule.shape[0]
        assert n == gain_schedule.shape[1]
    
    def dynamics(x,u,t):
        #print 'in dynamics'
        dxdt = np.dot(A,x)+np.dot(B,u)
        #print '     calc', dxdt
        return dxdt
    
    if(gain_schedule_ts is not None):
        #interpolate between samples of the gain_schedule
        #K = scipy.interpolate.interp1d(gain_schedule_ts,gain_schedule,axis=0)
        # the integrator likes to ask for values of K slightly larger than the 
        # horizon, so set a default value for the interpolator
        
        K = scipy.interpolate.interp1d(gain_schedule_ts,gain_schedule,axis=0,
                                       fill_value = gain_schedule[-1],
                                        bounds_error=False)
        
    else:
        def K(t):
            return gain_schedule
    
    def feedback(x,t):
        #print 'in feedback'
        #print '     K of',t
        assert t >= ts[0]
        return np.dot(K(t),x)
        
    dxdt = closed_loop_dynamics(dynamics,feedback)
    
    #should be equivalent     
    def dxdt1(x,t):
        return np.dot(A + np.dot(B,K(t)),x)
        
    traj = scipy.integrate.odeint(func=dxdt,y0=x0,t=ts)
    return traj
        
def simulate_lti_fb_dt(A,B,x0,gain_schedule,T):
    """
    gain_schdule is either one gain matrix or a sequence of gain matrices
    
    if a schedule:
    gain_schedule.shape = (T,m,n) array
    
    """
    gain_is_schedule = len(gain_schedule.shape)==3
    
    A = np.array(A)
    B = np.array(B)
    
    n = A.shape[0]
    m = B.shape[1]
    assert n == A.shape[1] == B.shape[0] 
    if gain_is_schedule:
        print 'schedule'
        assert m == gain_schedule.shape[1]
        assert n == gain_schedule.shape[2]
        assert T-1 == gain_schedule.shape[0]
    else:
        print 'not'
        assert len(gain_schedule.shape)==2
        assert m == gain_schedule.shape[0]
        assert n == gain_schedule.shape[1]
    
    xsol = np.zeros(shape=(n,T))
    usol = np.zeros(shape=(m,T-1))
    
    xsol[:,0] = x0
    
    for k in range(0,T-1):
        #print 'iter:',k        
        gk = gain_schedule[k,:,:] if gain_is_schedule else gain_schedule
        
        usol[:,k] = np.dot(gk,xsol[:,k])
        xsol[:,k+1] = np.dot(A,xsol[:,k]) + np.dot(B,usol[:,k])
    return xsol,usol
     
import cvxopt
import cvxopt.solvers

def LQR_QP(A,B,Q,R,T,x0,xT=None):
        """
        T number of time steps
        """
        n = A.shape[0]
        m = B.shape[1]
        assert n == A.shape[1] == B.shape[0] == Q.shape[0] == Q.shape[1]
        assert m == R.shape[0] ==R.shape[1]
        assert n == x0.size
        if(xT is not None):
            assert n == xT.size
        
        D = (n+m)*(T-1) + n #dimension of QP variables
        
        fvc = 0 if xT is None else 1
        Dq = (n)*(T+fvc) #dimension of equality constraints (number of equality constraints)
        """
        QP_var = [x[0],u[0],x[1],u[1],...,x[T-1],u[T-1],x[T]].T
        """
        QP_P = np.zeros(shape=(D,D))
        
        QP_A = np.zeros(shape=(Dq,D))
        QP_B = np.zeros(shape=(Dq,1))
        
        QP_q = np.zeros(shape=(D))

        for i in range(T):
            ul = i*(n+m)
            
            QP_P[ul:ul+n,ul:ul+n]=Q
            if i < T-1: #there is no control at the last time step
                QP_P[ul+n:ul+n+m,ul+n:ul+n+m]=R
        
        for i in range(T-1):
            r = (i+1)*n         #row start      
            c = i*(n+m)     #column start
            #dynamic constraint
            
            QP_A[r:r+n,c:c+n] = A #+ 10*np.ones_like(A)
            QP_A[r:r+n,c+n:c+n+m] = B #+ 20*np.ones_like(B)
            QP_A[r:r+n,c+n+m:c+n+m+n] = -np.eye(n)
            
        #initial value constraint
        QP_A[0:n,0:n] = np.eye(n)
        QP_B[0:n,0] = x0
        
        #final value constraint
        if xT is not None:
            QP_A[Dq-n:Dq,D-n:D] = np.eye(n)
            QP_B[Dq-n:Dq,0] = xT
        
        if(False):
            import matplotlib.pyplot as plt
            #plt.figure(None)
            #plt.title("P")
            #plt.spy(QP_P)
            #plt.imshow(QP_P,interpolation='nearest')
            
            plt.figure(None)
            plt.title("A")
            plt.spy(QP_A)
            plt.imshow(QP_A,interpolation='nearest')
        
        cQP_P = cvxopt.matrix(QP_P)
        cQP_q = cvxopt.matrix(QP_q)
        cQP_A = cvxopt.matrix(QP_A)
        cQP_B = cvxopt.matrix(QP_B)
                
        sol = cvxopt.solvers.coneqp(P=cQP_P,q=cQP_q,A=cQP_A,b=cQP_B)
        qp_sol = np.array(sol['x'],dtype=np.float32)    
        from numpy.lib.stride_tricks import as_strided
        
        dbyte = 4 #4 bytes in float32    
        xs = as_strided(qp_sol,shape=(n,T),strides=(dbyte,dbyte*(n+m)))
        us = as_strided(qp_sol[n:],shape=(m,T-1),strides=(dbyte,dbyte*(n+m)))
        return sol,(QP_P,QP_q,QP_A,QP_B),xs,us
            
            
            
            
                
        
        
        
    
    