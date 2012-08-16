

"""
forward system
x_{k+1} = A*x_k + B*u_k

A*x_k = x_{k+1} + (-B)*u_k 
x_k = A^{-1} * x_{k+1} + (-A^{-1}*B)*u_k 

Ar = A^{-1}
Br = -B*A^{-1}

if the forward system takes x0 to xT by applying u_0...u_{T-1}
then the reverse system takes xT to x0 by applying u_{T-1}...u_0

now how we need to use this:
LQR can find the policy that brings the system to xT in T time steps from anywhere -- call this ctg = LQR(A,B,xT,T). (ctg is cost-to-go)

What we want: find the cost of going from x0 to xT for a bunch of different Ts. One way: call LQR for each T and do x0^T * ctg * x0
Another way: 
ctgr = LQR(Ar,Br,x0,T)

"""

import examplets

from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR, final_value_LQR

import numpy as np
import matplotlib.pyplot as plt

#[velx, vely, posx, posy]
A = np.matrix([ [1,     0,      0,      0   ],
                [0,     1,      0,      0   ],
                [1e-1,  0,      1,      0   ],
                [0,     1e-1,   0,      1   ]])
#[accx, accy]
B = np.matrix([ [1,     0       ],
                [0,     1       ],
                [0,     0       ],
                [0,     0       ]])

Q = np.zeros(shape=(4,4))
R = np.eye(2)


Ar = A.I
Br = -A.I * B

def steer(A,B,x_from,x_toward):
    assert len(x_from) == 5
    T = x_toward[4] - x_from[4] #how much time to do the steering

    if T<=0:
        return (x_from,np.zeros(shape=(0,1)))   #stay here

    Fs, Ps = final_value_LQR(A,B,Q,R,x_toward[0:4],T)

    xs = np.zeros(shape=(T+1,5))
    us = np.zeros(shape=(T,2))
    xs[0] = x_from

    for i in range(T):
        us[i] = -1 * np.dot(Fs[i,:,0:4],xs[i,0:4]) + Fs[i,:,4]
        xs[i+1,0:4] = np.dot(A,xs[i,0:4].T) + np.dot(B,us[i].T)
        xs[i+1,4] = xs[i,4] + 1
    x_actual = xs[-1]    
    
    return (x_actual, us)

def run_forward(A,B,x0,us):
    n = 4+1  #+1 for time dimension 
    m = 2
    us = np.reshape(us,newshape=(-1,m))
    assert len(x0) == n
    T = us.shape[0]
    xs = np.zeros(shape=(T+1,n))
    xs[0] = np.squeeze(x0)

    for i in range(1,T+1):
        xs[i,0:4] = np.dot(A,xs[i-1,0:4].T) + np.dot(B,us[i-1].T)
        xs[i,4] = xs[i-1,4] + 1
    return xs[0:] #include x0 

def cost(A,B,x_from,action):
    #let U be the action trajectory
    #let T be the length of U.
    #let X be a trajectory  [x_from, A*X[0]+B*U[0], A*X[1]+B*U[1],... ] (the state trajectory gotten by applying U at x_from)
    #this calculates the sum X[i]'*Q*X[i] + U[i]'*Q*U[i] for i 0 to T
    #it does not include the term X[T+1]'*Q*X[T+1], where X[T+1] is the last element of X
    
    #this includes the cost of being in x_from if run_forward returns x_from
    #
    assert len(x_from) == 5
    assert action.shape[1] == 2
    x_path = run_forward(A,B,x_from,action)
    cost = 0
    for i in range(action.shape[0]):
        x = x_path[[i],0:4].T #don't include time
        u = action[[i],:].T
        cost += np.squeeze( np.dot(x.T,np.dot(Q,x))  + 
                            np.dot(u.T,np.dot(R,u))
                            ) 
    return cost



x0 = np.array([0,0,0,0,0])
xf = np.array([0,0,1,2,500])

_,us = steer(A,B,x0,xf)

#run the control trajectory forward on the forward system starting at x0
xs_forward = run_forward(A,B,x0,us)
#run the control trajectory backward on the backward system starting at xf and reverse the trajectory
xs_backward = run_forward(Ar,Br,xf,us[::-1])[::-1]

#xs_forward should equal xs_backward

plt.figure()
plt.subplot(3,1,1)
plt.plot(xs_forward[:,0:4])
plt.subplot(3,1,2)
plt.plot(xs_backward[:,0:4])
plt.subplot(3,1,3)
plt.plot(us)

#x0 and xf have the times embedded in them -- swap the times
xfr = np.array(xf)
x0r = np.array(x0)
xfr[4] = x0[4]
x0r[4] = xf[4]

#find the control trajectory that brings the reverse system from xfr to x0r
_,us_backward = steer(Ar,Br,xfr,x0r)
us_backward = us_backward[::-1]

xs_forward = run_forward(A,B,x0,us_backward)
xs_backward = run_forward(Ar,Br,xf,us_backward[::-1])[::-1]

plt.figure()
plt.subplot(3,1,1)
plt.plot(xs_forward[:,0:4])
plt.subplot(3,1,2)
plt.plot(xs_backward[:,0:4])
plt.subplot(3,1,3)
plt.plot(us)


T = xf[4]-x0[4] + 1
Fs, Ps = final_value_LQR(A,B,Q,R,xf[0:4],T)
#Ps[T] is the cost-to-go given zero time steps -- 
#Ps[T-1] is the cost-to-go given 1 time step
#Ps[0] is the cost-to-go given T time steps

x0m = x0[0:4].reshape(4,1)
x0m = np.vstack([x0m,[[1]]])

direct_cost_forward = cost(A,B,x0,us) #cost of taking x0 to xf
cost_to_go_forward = np.squeeze(np.dot(x0m.T,np.dot(Ps[0],x0m))) #same cost

#opposite 
Fsr, Psr = final_value_LQR(Ar,Br,Q,R,x0[0:4],T)

xfm = xf[0:4].reshape(4,1)
xfm = np.vstack([xfm,[[1]]])


direct_cost_backward =  cost(Ar,Br,xf,us[::-1]) #cost of taking xf to x0 in reverse system
cost_to_go_backward =  np.squeeze(np.dot(xfm.T,np.dot(Psr[0],xfm))) #same cost

l = [direct_cost_forward,cost_to_go_forward,direct_cost_backward,cost_to_go_backward]
print l
print 'std dev ', np.std(l)



