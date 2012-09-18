import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
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
from lin_ship_visualize_animation import Ship_Sprite
from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR
from lqr_rrt_cost import LQR_RRT
from rrt_interactive import RRT_Interactive

from lin_ship_cost import cost_zeroth_ord, cost_first_ord, cost_second_ord

#[velx, vely, posx, posy]
A = np.matrix([ [1,     0,      0,      0   ],
                [0,     1,      0,      0   ],
                [1,     0,      1,      0   ],
                [0,     1,      0,      1   ]])
#[accx, accy]
B = np.matrix([ [1,     0       ],
                [0,     1       ],
                [0,     0       ],
                [0,     0       ]])
    
max_time_horizon = 300
goal = np.array([0,0,100,100,max_time_horizon])

#field_shelve = shelve.open('field_simple.shelve')
#obstacle_paths = field_shelve['obstacle_paths']

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

from ship_field import obstacles_multipoly, obstacles_polys, get_patch_collection

#field_poly = Polygon(shell=[(-10,-10),(-10,110),(110,110),(110,-10),(-10,-10)],
#                    holes=[ [(0,0),(0,100),(100,100),(100,0),(0,0)] ] )

#obstacles_polys.append(field_poly)


ship_sprite = Ship_Sprite()


def isStateValid(state):
    #returns True if state is not in collision
    assert len(state) == 5
    #cheaper to check velocity
    if np.linalg.norm(state[[0,1]]) > 2: return False
    ship_sprite.update_pose(state[2],state[3])
    ship_poly = Polygon(ship_sprite.get_ship_path().vertices)
    return not ship_poly.intersects(obstacles_multipoly)

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

lqr_rrt = LQR_RRT(A,B,max_time_horizon,     cost_0 = cost_zeroth_ord, 
                                            cost_1 = cost_first_ord, 
                                            cost_2 = cost_second_ord )

rrt = RRT(state_ndim=5,control_ndim=2)

lqr_rrt.action_state_valid = action_state_valid

lqr_rrt.max_nodes_per_extension = 5

rrt.sample_goal = lambda : goal

rrt.set_distance(lqr_rrt.distance_cache)
rrt.set_same_state(lqr_rrt.same_state)
rrt.set_cost(lqr_rrt.cost)
rrt.set_steer(lqr_rrt.steer_cache)

rrt.set_goal_test(goal_test)
rrt.set_sample(sample)
rrt.set_collision_free(lqr_rrt.collision_free)

rrt.set_distance_from_goal(distance_from_goal)


rrt.gamma_rrt = 1.0
rrt.eta = .2
rrt.c = 1
rrt.max_nodes_in_ball = 30

#lqr_rrt.max_steer_cost = .015

rrt.set_start(start)
rrt.init_search()

def draw(rrt,ani_ax=None):
    if ani_ax is None:
        ani_ax = plt.figure().gca()

    ani_ax.cla()
    ani_ax.set_xlim(-10,110)
    ani_ax.set_ylim(-10,110)
    #ani_ax.set_aspect('equal')
    #ani_ax.set_aspect('auto')

    #should be able to move this out, but patch transforms get stuck and the obstacles don't pan/zoom
    obstacles_patches = [PolygonPatch(poly) for poly in obstacles_polys]
    obstacle_patch_collection = PatchCollection(obstacles_patches)    

    ani_ax.add_collection(obstacle_patch_collection)

    all_states = np.array(nx.get_node_attributes(rrt.tree,'state').values())
    ani_ax.plot(all_states[:,2],all_states[:,3],'g.',alpha=.8,zorder=2)
    
    import copy
    if False:
        for state in all_states:
            ship_sprite.update_pose(state[2],state[3],0)
            ship_sprite.update_transform_axes(ani_ax)
            for patch in ship_sprite.patches:
                ani_ax.add_artist(copy.copy(patch))
    
    #draw dynamical edges
    lines = []
    for i in rrt.tree.nodes():
        s = rrt.tree.predecessors(i)
        if len(s) == 0:
            continue
        assert len(s) == 1 #it's a tree
        s = s[0]
        x0 = rrt.tree.node[s]['state']
        xs = lqr_rrt.run_forward(x0, rrt.tree.node[i]['action'])
        xs = np.concatenate((x0.reshape((1,-1)),xs))
        lines.append(xs[:,[2,3]])
    edge_collection = mpl.collections.LineCollection(lines)
    ani_ax.add_collection(edge_collection)
    
    a = rrt.best_solution_goal()
    if a is not None:
        nodes, xpath_sparse, upath = a
        xpath = lqr_rrt.run_forward(start,upath)
        ani_ax.plot(xpath[:,2],xpath[:,3],'.',zorder=3)

def hook(rrt):
    plt.ioff()
    a = plt.figure()

    c = rrt.worst_cost
    fname = "rrt_2d_di_%d,%d.png"%(start_time,rrt.n_iters)
    draw(rrt,a.gca())
    a.savefig(fname)
    plt.ion()

    import shelve
    s = shelve.open("rrt_2d_di_%d,%d.shelve"%(start_time,rrt.n_iters))
    #upath = rrt.best_solution_goal()[2]
    #xpath = lqr_rrt.run_forward(start,upath)
    #s['traj'] = xpath
    #s['utraj'] = upath
    print 'saving {}'.format(rrt.n_iters)
    rrt.save(s)
    s.close()


def generate_partial_trees(rrt):
    sample_goal = rrt.sample_goal
    rrt.sample_goal = None
    
    for i in [50,500,2000]:
        rrt.search(i)
        hook(rrt)
    
rrt.improved_solution_hook = hook

rrt_int = RRT_Interactive(rrt,lqr_rrt.run_forward,plot_dims=[2,3],slider_range=(0,max_time_horizon))

obstacles_patches = [PolygonPatch(poly) for poly in obstacles_polys]
obstacle_patch_collection = PatchCollection(obstacles_patches)    
rrt_int.int_ax.add_collection(obstacle_patch_collection)

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
    
    
