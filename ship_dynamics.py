# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 04:23:47 2012

@author: gustavo
"""

import sympy

from sympy.utilities.autowrap import autowrap
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
theta = sympy.Symbol('theta')

vx = sympy.Symbol('vx')
vy = sympy.Symbol('vy')
vtheta = sympy.Symbol('vtheta')
dt = sympy.Symbol('dt')
lin_thrust = sympy.Symbol('lin_thrust')
ang_thrust = sympy.Symbol('ang_thrust')

state_symbol = sympy.Matrix([vx,vy,vtheta,x,y,theta])
control_symbol = sympy.Matrix([lin_thrust,ang_thrust])
parameter_symbol = [dt]
parameter_bindings = {dt:5e-2}

n = len(state_symbol)
m = len(control_symbol)

dynamics_symbols = list(state_symbol) + list(control_symbol)

symbolic_dynamics = sympy.Matrix([vx + dt * sympy.cos(theta) * lin_thrust,
                                  vy + dt * sympy.sin(theta) * lin_thrust,
                                  vtheta + dt * ang_thrust,
                                  x + dt * vx,
                                  y + dt * vy,
                                  theta + dt * vtheta])

assert len(symbolic_dynamics) == n

symbolic_dynamics_p = symbolic_dynamics.subs(parameter_bindings)

#produces a list of functions
dyn_f_list = [autowrap(symbolic_dynamics_p[i],language="f95",args=dynamics_symbols) for i in range(len(symbolic_dynamics))]

def dyn_f(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)
    return np.array(
        [dyn_f_list[i](*xu) for i in range(len(symbolic_dynamics_p))]
    )


linearized_sym_A = symbolic_dynamics_p.jacobian(state_symbol)
linearized_sym_B = symbolic_dynamics_p.jacobian(control_symbol)

lin_A_aw = [
                [
                    autowrap(linearized_sym_A[i,j],language="f95",args=dynamics_symbols)
                for i in range(n)] 
            for j in range(n)]

lin_B_aw = [
                [
                    autowrap(linearized_sym_B[i,j],language="f95",args=dynamics_symbols)
                for i in range(n)] 
            for j in range(m)]

def linearized_A(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)

    return np.array(
        [
            [lin_A_aw[i][j](*xu) for i in range(n)] 
        for j in range(n)]
    )

def linearized_B(x,u):
    assert len(x)==n
    assert len(u)==m
    xu = list(x) + list(u)

    return np.array(
        [[lin_B_aw[i][j](*xu) for i in range(n)] for j in range(m)]
    )

if __name__ == 'main':
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





