# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:42:53 2012

@author: gustavo
"""

"""

Purpose is to show that using cost-to-go matrix S 
or its inverse P in the Riccati iteration gives same solution
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


def AQR(A,B,c,Q,q,R,r):
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
    print 'shape',c.shape
    assert n == c.shape[0] == q.shape[0]
    
    nh = n + 1
    
    Ah = np.zeros(shape=(nh,nh))
    Bh = np.zeros(shape=(nh,m))
    Qh = np.zeros(shape=(nh,nh))

    
    Ah[0:n,0:n] = A
    test = c - B*R.I*r
    Ah[0:n,n] = test[:].T
    Ah[n,n]=1
    
    Bh[0:n,:] = B
    
    Qh[0:n,0:n]=Q
    Qh[n,0:n] = q.T
    Qh[0:n,n] = q.T
    Qh[n,n] = 10 #arbitrary
    
    Rh = R
    
    return (Ah,Bh,Qh,Rh,-R.I*r)
    
def ftdlqr(A,B,Q,R,N,Q_terminal=None):
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
        
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k])
        Ps[k-1]=Q + A.T * (Pk - Pk*B*(R+B.T*Pk*B).I * B.T*Pk)*A
    
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k])      
        Fs[k] = (R+B.T*Pk*B).I * B.T * Pk * A

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

def ftdlqr_dual(A,B,Q,R,N,Q_terminal_inv=None):
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
        
    
    for k in range(N-1,0,-1):
        Pk = np.matrix(Ps[k]).I      
        Fs[k] = (R+B.T*Pk*B).I * B.T * Pk * A
        
    return (Fs,Ps)

    
    
def rot_2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

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


(Ah,Bh,Qh,Rh,pk) = AQR(A=A,B=B,c=c,Q=Q,R=R,q=q,r=r)

n = 50
x0 = np.array([0e0,0])

DUAL = True
if not DUAL:
    gain_matrices,cost_to_go_matrices = ftdlqr(Ah,Bh,Qh,Rh,N=n,
                                               Q_terminal=Qh*1e0)
else:
    gain_matrices,cost_to_go_matrices = ftdlqr_dual(Ah,Bh,Qh,Rh,N=n,
                                        Q_terminal_inv=np.linalg.inv(Qh*1e0))

dsol = np.matrix(np.zeros(shape=(2,n)))
dcontrol = np.matrix(np.zeros(shape=(1,n)))
dsol[:,0] = np.matrix(x0).T

cost_to_go = np.zeros(shape=(n))

for k in range(0,n-1):
    print 'iter:',k

    errk = dsol[:,k] 
        
    gk = gain_matrices[n-k-1,:,:]
    #gk = gain_matrices[-1,:,:]
    #both compute same thing
    uk = -np.dot(gk[:,0:2],errk) + np.sum(gk[:,2])    
    uk = -np.dot(gk[:,:],np.concatenate((errk,[[1]])))
    #print uk
    dcontrol[:,k+1] = uk
    #uk =0
    dsol[:,k+1] = np.dot(A,dsol[:,k]) + np.dot(B,uk) + c
       
dsol1 = np.vstack( (dsol,np.ones(shape=(1,n))) )


for k in range(0,n):
    if DUAL:
        ctg = np.linalg.inv(cost_to_go_matrices[k])
    else:
        ctg = cost_to_go_matrices[k]
        
    a = np.dot(ctg,dsol1[:,k])
    cost_to_go[k] = np.dot(dsol1[:,k].T,a)

plt.subplot(2,1,1)
plt.plot(dsol[0,:].T,'g-') #velocity
plt.plot(dsol[1,:].T,'b-') #position
plt.plot(dcontrol[0,:].T,'r-') #imparted acceleration
plt.axhline(color='k')

plt.subplot(2,1,2)
plt.plot(cost_to_go,'g-')
plt.axhline(color='k')
plt.show()
    
    
