"""
demonstrates how changing time horizon changes cost non-monotonically.

In this example, the dynamical system is a second-order oscilator (spring-mass-dampener system)

the cost-to-go will be zero if the system will be at the desired point, just by following the dynamics, at the right time.
since the dynamics are periodic, there are multiple time-horizons that produce a zero cost-to-go, namely all time horizons which are multiples of the period of the dynamics.

You get similar results if the dynamics are pseudo-periodic (add dampening d =.5)

with an actuation cost:
 R = np.eye(1)*1e0 

and dynamics:
d=  .05       #damping
k  = 1      #spring

the benefit of a longer horizon (and therefore being able to apply smaller controls) approximately equals the energy loss from dampening. 

for larger damping values, longer horizons trend upward in cost. 
for smaller damping values, longer hoirzons trend downward.

QUESTION:
the "period" of the cost-to-go seems to be half that of the dynamic system. Why?
(period_samp/cost_to_go_period = 2)

"""


import lqr_tools
import yottalab
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

#continuous-time dynamics. 
d=  0       #damping

k  = 1      #spring

Act = np.matrix([[d,-k],
                 [1,0]])


Bct = np.matrix([[1],
                 [0]])

eigenvalue = np.linalg.eig(Act)[0][0]
period_sec = 2* np.pi/np.abs(np.imag(eigenvalue)) #period in seconds

n=2 #state dim
nb=1 #control dim
Ts = 1e-1

period_samp = period_sec/Ts #period in samples
#calculate discreet time dynamics
a = Act
b = Bct
ztmp=np.zeros((nb,n+nb))
tmp=np.hstack((a,b))
tmp=np.vstack((tmp,ztmp))
tmp=expm(tmp*Ts)
A=tmp[0:n,0:n]
B=tmp[0:n,n:n+nb]
              

#only penalize error (deviation from desired) at the final time step.
Q = np.matrix([[0,0],
               [0,0]])

#to demonstate that there are, in general, multiple minima in the cost-to-go function, must assign unequal weights
#to the state variables.
Q_terminal = np.matrix([[0,0],
                        [0,1]])

R = np.eye(1)*1e0

#Fs are gain matrices, Ps are cost-to-go matrices
(Fs, Ps) = lqr_tools.dtfh_lqr(A,B,Q,R,Q_terminal = Q_terminal,N=200)

#the set point
desired = np.matrix([[0],
                     [1]])

#starting from 
initial = np.matrix([ [0],
                      [0]])

error = initial-desired
#compute series of cost-to-go values starting from . the following just computes (error^T * P_i * error) for all i
""" unvectorized:
b = []
for P in Ps:
    b.append( np.dot(desired.T,np.dot(P,desired))[0,0] )
b = np.array(b)
"""
a = np.tensordot(Ps,error,axes=([2],[0]))
b = np.tensordot(a,error,axes=([1],[0]))
b = b.flatten()

fig = plt.gcf()

ax_ctg = fig.add_subplot(2,1,1)
ax_ctg.set_ylabel('cost-to-go')
ax_ctg.plot(b[::-1])

ax_dctg = fig.add_subplot(2,1,2,sharex=ax_ctg)
ax_dctg.set_ylabel('deriative of cost-to-go w.r.t. horizon')
ax_dctg.plot(np.diff(b)[::-1])
ax_dctg.set_xlabel('horizon length')

plt.show()
print 'Q eig:',np.linalg.eig(Q)


#approximate the period of the cost-to-go function as the time horizon changes

#the first term is an array of bools true where the sequence is decreasing
#the second term is an array of bools for where the sequence is increasing
mins = np.concatenate([[True], b[1:] < b[:-1]]) & np.concatenate([b[:-1] < b[1:], [True]])

min_indx = np.where(mins==True)[0]
diffs = np.diff(min_indx[1:]) #ignore first 'minimum' due to boundary effects
cost_to_go_period = np.mean(diffs)

print "cost-to-go period: %f, dynamics period: %f , factor: %f"%(cost_to_go_period,period_samp,period_samp/cost_to_go_period)



