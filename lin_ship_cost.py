# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 04:23:47 2012

@author: gustavo
"""

import sympy

from sympy.core import numbers

import sympy.functions

from sympy.utilities.autowrap import autowrap
from mod_autowrap import autowrap_and_store 
import functools

autowrap = functools.partial(autowrap,verbose=True,tempdir='.autowrap_cache')
#autowrap = autowrap_and_store


real_autowrap = autowrap

def lazy_autowrap(*args,**kwargs):
    expr = args[0]
    if isinstance(expr,numbers.Number):
        print "autowrap constant expression"
        def dummy(*args):
            assert len(args) == len(kwargs['args'])
            return expr.evalf()
        return dummy
    else:
        return real_autowrap(*args,**kwargs)

autowrap = lazy_autowrap
import numpy as np

import shelve

"""
state: [x,y,theta,vx,vy,vtheta]

CT:
d/dt(vx) = cos(theta) * for_thrust
d/dt(vy) = sin(theta) * for_thrust
d/dt(vtheta) = rot_thrust

d/dt(x) = vx
d/dt(y) = vy
d/dt(theta) = vtheta

vx = vx + dt * (cos(theta) * for_thrust)
vy = vy + dt * (sin(theta) * for_thrust)
vtheta = vtheta + rot_thrust

x = x + dt * vx
y = y + dt * vy
theta = theta + dt * vtheta

"""

x = sympy.Symbol('x')
y = sympy.Symbol('y')


vx = sympy.Symbol('vx')
vy = sympy.Symbol('vy')

dt = sympy.Symbol('dt')
x_thrust = sympy.Symbol('x_thrust')
y_thrust = sympy.Symbol('y_thrust')

state_symbol = sympy.Matrix([vx,vy,x,y])
control_symbol = sympy.Matrix([x_thrust,y_thrust])
parameter_symbol = [dt]
parameter_bindings = {dt:5e-2}

n = len(state_symbol)
m = len(control_symbol)

dynamics_symbols = list(state_symbol) + list(control_symbol)



x0 = sympy.Symbol('x0')
y0 = sympy.Symbol('y0')

vx0 = sympy.Symbol('vx0')
vy0 = sympy.Symbol('vy0')


x_thrust0 = sympy.Symbol('x_thrust0')
y_thrust0 = sympy.Symbol('y_thrust0')

xu0 = sympy.Matrix([vx0,vy0,x0,y0,x_thrust0,y_thrust0])
xu_star = sympy.Matrix(dynamics_symbols) - xu0

cost =  sympy.functions.exp(y/3)*(x_thrust**2) + y_thrust**2


#first-order coef
cost_grad = sympy.Matrix([[cost]]).jacobian(dynamics_symbols)

#q = cost_grad[:n]
#r = cost_grad[n:]

#second-order coef
cost_hes = sympy.hessian(cost,dynamics_symbols)
#Q = cost_hes[:n,:n]
#R = cost_hes[n:,n:]
#assert R.shape == (m,m)


#Symbolic Taylor expansion
if False:
    mapping = {x:x0,y:y0,vx:vx0,vy:vy0,x_thrust:x_thrust0,y_thrust:y_thrust0}
    grad0 = cost_grad.subs(mapping)
    hes0 = cost_hes.subs(mapping)

    approx = cost.subs(mapping)+ (grad0*xu_star)[0,0] + (xu_star.T*hes0*xu_star)[0,0]/2
    approx = approx.simplify()


print 'compiling cost Jacobians'
cost_aw = autowrap(cost,language="f95",args=dynamics_symbols)

cost_grad_aw = [autowrap(cost_grad[i],language="f95",args=dynamics_symbols)
                for i in range(n+m)]


cost_hessian_aw =   [
                        [
                            autowrap(cost_hes[i,j],language="f95",args=dynamics_symbols)
                        for j in range(n+m)] 
                    for i in range(n+m)]
print 'done compiling cost Jacobians'

#coefficients for Taylor approximation
def cost_zeroth_ord(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)
    return cost_aw(*xu)

def cost_first_ord(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)
    return np.array([float(cost_grad_aw[i](*xu)) for i in range(n+m)]).reshape((1,n+m))

def cost_second_ord(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)

    return np.array(
        [[float(cost_hessian_aw[i][j](*xu)) for i in range(n+m)] for j in range(n+m)]
    )

if __name__ == 'main':
    print 'running main'
    T = 1000
    state0 = np.array([0,0,0,0,0,0])
    traj = np.zeros(shape=(T,n))
    traj[0] = state0

    utraj = np.zeros(shape=(T,m))
    utraj[0:100,0] = 1
    utraj[500:550,0] = 1
    utraj[650:700,0] = 10
    utraj[700:800,0] = -5

    #rot
    utraj[500:600,1] = 1e-1
    utraj[600:610,1] = -10e-1
    utraj[800:810,1] = -10e-1
    utraj[860:870,1] = 10e-1

    #simulate dynamics
    for i in range(T-1):
        traj[i+1] = dyn_f(traj[i],utraj[i])

    ship_shelve = shelve.open('ship.shelve')
    ship_shelve['T']=T
    ship_shelve['traj']=traj
    ship_shelve['utraj']=utraj
    ship_shelve.close()





