import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#only to label 
import time
start_time = int(time.time())

import sys
sys.setrecursionlimit(10000)

import shelve
import itertools
import networkx as nx

from rrt import RRT
from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR
from diff_rrt import Diff_RRT
from rrt_interactive import RRT_Interactive

"""
theta = 0 means pendulum is pointing down
theta > 0 means pendulum has rotated in the counter-clockwise direction
thetadot > 0 means pendulum is rotating counter-clockwise

nonlinear:
thetadot = (1-damping) * thetadot + dt*(-sin(theta) + u) 
theta = theta + dt*theta_dot

linear:
(sin(x) =approx= cos(x0) * (x-x0) + x0 ) 

thetadot = thetadot + dt * (-cos(theta0)*(theta-theta0) +theta0)  + dt *u
theta = theta + dt*theta_dot

thetadot = thetadot + dt * -cos(theta0)*theta + dt*theta0 * (cos(theta0) +1)  + dt *u


x = Ax + Bu + c

A = [[1-damping ,-cos(theta0)*dt], [dt, 1]]
B = [[dt] , [0]]
c = [[dt*theta0 * (cos(theta0) +1) ] , [0]]



A(x-x0) + B(u-u0) +f(x,u) = Ax+Bu - A*x0 - B*u0 + f(x,u)

-A*x0 = [[thetadot0 - cos(theta0)*dt*theta0] , [thetadot0*dt + theta0]

f(x0)= [[thetadot0 + dt * (-sin(theta0) + u0)],[theta0 + dt*thetadot0]]
"""

dt = .05
damping = .005

def linearized_A(x,u):
    A = np.zeros((2,2))
    A[0,0] = 1 - damping
    A[1,0] = dt
    A[1,1] = 1
    A[0,1]= -np.cos(x[1]) * dt
    return A

def linearized_B(x,u):
    B = np.zeros((2,1))
    B[0] = dt
    return B

def dyn_f(x,u):
    xk0 = (1.0-damping) * x[0] + dt*(-np.sin(x[1]) + u[0])
    xk1 = ( x[1] + dt*x[0] ) * np.ones_like(xk0) #make have same dimension due to u
    xk = np.concatenate((xk0,xk1),axis=0)
    return xk

def cost_0(x,u):
    return u[0]**2

def cost_1(x,u):
    return np.zeros((1,3))

def cost_2(x,u):
    hes = np.zeros((3,3))
    hes[2,2] = 1
    return hes

def get_ABc(x,u):
    A = linearized_A(x,u)
    B = linearized_B(x,u)
    c = np.zeroes((2,1))

    c[0] = dt(x[1] * np.cos(x[1] - np.sin(x[1]) ))

    return A,B,c

def get_QRqrd(x,u):
    Q = np.zeros((2,2))
    R = np.ones((1,1))
    q = np.zeros((2,))
    r = np.zeros((1))
    d = 0
    return Q,R,q,r,d    
            
max_time_horizon = 700
goal = np.array([0,np.pi,500])

def isStateValid(state):
    #returns True if state is not in collision
    assert len(state) == 3
    return True

def isActionValid(action):
    assert len(action) == 1
    return np.linalg.norm(action) < .4

def action_state_valid(x,u):
    return isStateValid(x) and isActionValid(u)

def sample():
    if np.random.rand()<.9:
        statespace = np.random.rand(2)*np.array([6,2*np.pi])-np.array([3,np.pi])
        time = np.random.randint(0,max_time_horizon,size=1) + 1
        #time = np.array(min(np.random.geometric(.06,size=1),max_time_horizon))
        time = np.reshape(time,newshape=(1,))
        return np.concatenate((statespace,time))
    else: #goal bias
        statespace = goal[0:2]
        time = np.random.randint(0,max_time_horizon,size=1) + 1
        #time = np.array(min(np.random.geometric(.06,size=1),max_time_horizon))
        time = np.reshape(time,newshape=(1,))
        return np.concatenate((statespace,time))
        

def distance_from_goal(node):
    return 0

def goal_test(node):
    goal_region_radius = .01
    n = 4
    return np.sum(np.abs(node['state'][0:n]-goal[0:n])) < goal_region_radius #disregards time
    return distance(node,goal) < goal_region_radius                     #need to think more carefully about this one

start = np.array([0,0,0])

lqr_rrt = Diff_RRT( linA = linearized_A,
                    linB = linearized_B,
                    dyn_f= dyn_f,
                    cost_0 = cost_0, 
                    cost_1 = cost_1, 
                    cost_2 = cost_2, 
                    max_time_horizon = max_time_horizon,
                    n=2,
                    m=1
                  )

rrt = RRT(state_ndim=3,control_ndim=1)

lqr_rrt.action_state_valid = action_state_valid
lqr_rrt.max_nodes_per_extension = 5

rrt.sample_goal = lambda : goal

rrt.set_distance(lqr_rrt.distance_cache)
rrt.set_same_state(lqr_rrt.same_state)
rrt.set_cost(lqr_rrt.cost)
#rrt.set_steer(lqr_rrt.steer_cache)
rrt.set_steer(lqr_rrt.steer)

rrt.set_goal_test(goal_test)
rrt.set_sample(sample)
rrt.set_collision_free(lqr_rrt.collision_free)

rrt.set_distance_from_goal(distance_from_goal)

rrt.gamma_rrt = 15
rrt.eta = 20
rrt.c = 1
rrt.max_nodes_in_ball = 30

rrt.set_start(start)
rrt.init_search()



def draw(rrt,ani_ax=None):
    if ani_ax is None:
        ani_ax = plt.figure().gca()

    
    ani_ax.cla()
    ani_ax.set_xlim(-10,10)
    ani_ax.set_ylim(-np.pi,np.pi)
    #ani_ax.set_aspect('equal')
    #ani_ax.set_aspect('auto')

    #ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-10,110,-10,110],alpha=.5,zorder=1,aspect='auto')    

    all_states = np.array(nx.get_node_attributes(rrt.tree,'state').values())
    ani_ax.plot(all_states[:,0],all_states[:,1],'g.',alpha=.8,zorder=2)


    a = rrt.best_solution_goal()
    if a is not None:
        nodes, xpath_sparse, upath = a
        xpath = lqr_rrt.run_forward(start,upath)
        ani_ax.plot(xpath[:,0],xpath[:,1],'.',zorder=3)

    tree = rrt.tree
    #draw dynamical edges
    lines = []
    for i in tree.nodes():
        s = tree.predecessors(i)
        if len(s) == 0:
            continue
        assert len(s) == 1 #it's a tree
        s = s[0]
        x0 = tree.node[s]['state']
        xs = lqr_rrt.run_forward(x0, tree.node[i]['action'])
        xs = np.concatenate((x0.reshape((1,-1)),xs))
        lines.append(xs[:,[0,1]])
    edge_collection = mpl.collections.LineCollection(lines,alpha=.5,zorder=0)
    ani_ax.add_collection(edge_collection)
    


def generate_partial_trees(rrt):
    sample_goal = rrt.sample_goal
    rrt.sample_goal = None
    
    for i in [20,50,100,300,300]:
        if i>50: rrt.sample_goal = sample_goal #turn off goal bias to get exploration in the whole space
        rrt.search(i)
        hook(rrt)

if __name__ == '__main__':

    def hook(rrt):
        print 'hook'
        plt.ioff()
        a = plt.figure()
        c = rrt.worst_cost
        print 'draw'
        fname = "rrt_pendulum__%d,%d.png"%(start_time,rrt.n_iters)
        draw(rrt,a.gca())
        a.savefig(fname)
        plt.ion()

        print 'save'
        import shelve
        s = shelve.open("rrt_pendulum_%d,%d.shelve"%(start_time,rrt.n_iters))
        #upath = rrt.best_solution_goal()[2]
        #xpath = lqr_rrt.run_forward(start,upath)
        #s['traj'] = xpath
        #s['utraj'] = upath
        rrt.save(s)
        print 'saved {}'.format(rrt.n_iters)
        s.close()


    rrt.improved_solution_hook = hook

    isinteractive = plt.isinteractive()

    if isinteractive: plt.ioff()
    rrt_int = RRT_Interactive(rrt,lqr_rrt.run_forward,plot_dims=[0,1],slider_range=(0,max_time_horizon))
    rrt_int.int_ax.set_xlim(-3,3)
    rrt_int.int_ax.set_ylim(-5,5)
    rrt_int.rrts()



if False and __name__ == '__main__':
#    if False:
#        rrt.load(shelve.open('kin_rrt.shelve'))

    i = 0
    if i>0:
        rrt.load(shelve.open('linship_rrt_%04d.shelve'%(i-1)))
  
    while (not rrt.found_feasible_solution):
        rrt.search(iters=5e1)
        s = shelve.open('linship_rrt_%04d.shelve'%i)
        rrt.save(s)
        s.close()
        i+=1
        #nearest_id,nearest_distance = rrt.nearest_neighbor(goal)
        #print 'nearest neighbor distance: %f, cost: %f'%(nearest_distance,rrt.tree.node[nearest_id]['cost'])
        

    rrt.search(iters=5e2)
    xpath = np.array([rrt.tree.node[i]['state'] for i in rrt.best_solution(goal)]).T

    T = xpath.shape[1]
    traj = np.zeros((T,4))
    utraj = np.zeros((T,2))
    
    
