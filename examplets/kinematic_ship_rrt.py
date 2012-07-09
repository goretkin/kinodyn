import examplets
from rrt import RRT
from ship_visualize_animation import Ship_Sprite

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import networkx as nx

import sys
sys.setrecursionlimit(10000)

import shelve

field_shelve = shelve.open('field1.shelve')
obstacle_paths = field_shelve['obstacle_paths']

ship_sprite = Ship_Sprite()

def collision_check(state):
    #returns True if state is not in collision
    ship_sprite.update_pose(*state)
    does_collide = ship_sprite.collision2(obstacle_paths)
    return not does_collide

#dummy test
#def collision_check(state):
#    return True

goal = np.array([100.0,100.0,0])

def sample():
    global goal
    if np.random.rand()<.8:
        r = np.random.rand(3)
        r[0] = r[0]*120-10
        r[1] = r[1]*120-10
        r[2] = r[2]*2*np.pi
        return r
    else: #goal bias
        return goal

def collision_free(from_node,to_point):
    """
    check that the line in between is free
    """
    #print from_node,to_point,'collision_free'
    (x,y,t) = from_node['state']
    (x1,y1,t1) = to_point
    if not collision_check((x,y,t)):
        return [],False
    direction = to_point-from_node['state']
    l = np.linalg.norm(direction)
    direction /= l
    step = .5
    
    free_points = []
    all_the_way = True
    for i in range(int(l/step)):
        inter = direction*step*(i+1)+from_node['state']
        if( collision_check(inter)):
            free_points.append(np.array(inter))
        else:
            all_the_way = False
            break
    
    #add the to_point if it's not in an obstacle and the line was free
    if all_the_way and collision_check(to_point):
        free_points.append(np.array(to_point))
    else:
        all_the_way = False
    return free_points, all_the_way    

maximum_extension = 30
def steer(x_from,x_toward):
    extension_direction = x_toward-x_from
    
    if abs(extension_direction[2]) > np.pi:
        #go the other way
        extension_direction[2] = extension_direction[2] - 2*np.pi*np.sign(extension_direction[2])
    
    norm = np.linalg.norm(extension_direction)
    if norm > maximum_extension:
        extension_direction = extension_direction/norm
        extension_direction *= maximum_extension
    control = extension_direction #steer
    
    x_new = x_from + control 
    return (x_new,control)
    
def distance(from_node,to_point):
    #to_point is an array and from_point is a node
    delta = to_point-from_node['state']
    
    if abs(delta[2]) > np.pi:
        #go the other way
        delta[2] = delta[2] - 2*np.pi*np.sign(delta[2])
    delta[2] *= 10 #penalize turning
    return np.linalg.norm(delta)

goal_region_radius = 1e-2

def goal_test(node):
    global goal
    return distance(node,goal) < goal_region_radius

def distance_from_goal(node):
    global goal
    return max(distance(node,goal)-goal_region_radius,0)

start = np.array([0,0,0])
goal = np.array([100.0,100.0,0])

rrt = RRT(state_ndim=3,keep_pruned_edges=False)

rrt.set_distance(distance)
rrt.set_steer(steer)

rrt.set_goal_test(goal_test)
rrt.set_distance_from_goal(distance_from_goal)

rrt.set_sample(sample)
rrt.set_collision_check(collision_check)
rrt.set_collision_free(collision_free)

rrt.gamma_rrt = 100.0
rrt.eta = 50.0
rrt.c = 1

rrt.set_start(start)
rrt.init_search()



if __name__ == 'main':
    if False:
        rrt.load(shelve.open('kin_rrt.shelve'))



    while (not rrt.found_feasible_solution):
        rrt.search(iters=5e1)
        nearest_id,nearest_distance = rrt.nearest_neighbor(goal)
        print 'nearest neighbor distance: %f, cost: %f'%(nearest_distance,rrt.tree.node[nearest_id]['cost'])

    s = shelve.open('kin_rrt.shelve')
    rrt.save(s)
    s.close()

    s = shelve.open('kin_rrt.shelve')
    assert set(s.keys()) == set(rrt.save_vars)
    s.close()

    rrt.search(iters=5e3)
    xpath = np.array([rrt.tree.node[i]['state'] for i in rrt.best_solution(goal)]).T

    T = xpath.shape[1]
    traj = np.zeros((T,6))
    utraj = np.zeros((T,2))

    traj[:,3:6] = xpath.T



    s = shelve.open('kin_traj.shelve')
    s['T'] = T
    s['utraj'] = utraj
    s['traj'] = traj
    s.close()

