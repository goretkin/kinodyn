#import examplets
from rrt import RRT
from ship_visualize_animation import Ship_Sprite

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import networkx as nx

import sys
sys.setrecursionlimit(10000)

import shelve

import itertools
import networkx as nx

from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR

from lqr_rrt import LQR_RRT
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

max_time_horizon = 200
goal = np.array([0,0,100,100,200])


#field_shelve = shelve.open('field_simple.shelve')
#obstacle_paths = field_shelve['obstacle_paths']

ship_sprite = Ship_Sprite()

def obstacle(x,y):
    u = (x+y)/2
    v = x-y

    #in_obstacle1 = np.logical_and(np.logical_and(20<=x,x<=55),np.logical_and(45<=y,y<=80))
    #in_obstacle2 = np.logical_and(np.logical_and(70<=x,x<=110),np.logical_and(60<=y,y<=80))

    in_obstacle1 = np.logical_and(np.logical_and(45<=u,u<=75),np.logical_and(-25<=v,v<=-5))
    in_obstacle2 = np.logical_and(np.logical_and(45<=u,u<=75),np.logical_and(5<=v,v<=25))

    in_field = np.logical_and(np.logical_and(-10<=x,x<=110),np.logical_and(-10<=y,y<=110))        
    return np.logical_and ( np.logical_not( np.logical_or(in_obstacle1,in_obstacle2) ),
                            in_field
                            )

def isStateValid(state):
    #returns True if state is not in collision
    assert len(state) == 5
    #ship_sprite.update_pose(state[2],state[3],0)
    #does_collide = ship_sprite.collision2(obstacle_paths)
    within_vel = np.linalg.norm(state[0:2]) < 20 #velocity limit

    return within_vel and obstacle(state[2],state[3])

def isActionValid(action):
    assert len(action) == 2
    #np.linalg.norm(action) < .24
    return True 

def action_state_valid(x,u):
    return isStateValid(x) and isActionValid(u)

def sample():
    if np.random.rand()<.9:
        statespace = np.random.rand(4)*np.array([10,10,120,120])-np.array([5,5,10,10])
        time = np.random.randint(0,max_time_horizon,size=1) + 1
        #time = np.array(min(np.random.geometric(.06,size=1),max_time_horizon))
        time = np.reshape(time,newshape=(1,))
        return np.concatenate((statespace,time))
    else: #goal bias
        statespace = goal[0:4]
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


start = np.array([0,0,0,0,0])
lqr_rrt = LQR_RRT(A,B,Q,R,max_time_horizon)
rrt = RRT(state_ndim=5,control_ndim=2)

lqr_rrt.action_state_valid = action_state_valid


rrt.sample_goal = lambda : goal

rrt.set_distance(lqr_rrt.distance_cache)
rrt.set_same_state(lqr_rrt.same_state)
rrt.set_cost(lqr_rrt.cost)
rrt.set_steer(lqr_rrt.steer_cache)

rrt.set_goal_test(goal_test)
rrt.set_sample(sample)
rrt.set_collision_free(lqr_rrt.collision_free)

rrt.set_distance_from_goal(distance_from_goal)


rrt.gamma_rrt = 5
rrt.eta = .5
rrt.c = 1

rrt.set_start(start)
rrt.init_search()

def draw():
    x = np.linspace(-10,110,500)
    X,Y = np.meshgrid(x,x)
    obstacle_bitmap = obstacle(X,Y) #rasterize the obstacles

    ani_ax = plt.figure().gca()

    
    ani_ax.cla()
    ani_ax.set_xlim(-10,110)
    ani_ax.set_ylim(-10,110)
    #ani_ax.set_aspect('equal')
    #ani_ax.set_aspect('auto')

    ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-10,110,-10,110],alpha=.5,zorder=1,aspect='auto')    

    all_states = np.array(nx.get_node_attributes(rrt.tree,'state').values())
    ani_ax.plot(all_states[:,2],all_states[:,3],'g.',alpha=.8,zorder=2)


    a = rrt.best_solution_goal()
    if a is None: return
    nodes, xpath_sparse, upath = a
    xpath = lqr_rrt.run_forward(start,upath)
    ani_ax.plot(xpath[:,2],xpath[:,3],'.',zorder=3)



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
    
    
