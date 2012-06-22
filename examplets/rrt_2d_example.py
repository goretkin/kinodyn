# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:55:29 2012

@author: gustavo
"""

import examplets
from rrt import RRT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

def obstacles(x,y):
    #return true for if point is not in collision
    out_ball = (x-0)**2 + (y-0)**2 > .5**2
    in_square = np.logical_and(y>.1,np.logical_and(.3<x,x<.5))
    in_square1 = np.logical_and(np.logical_and(.6<x,x<.99),np.logical_and(.4<y,y<.8))
    
    return np.logical_and(out_ball, 
                          np.logical_and(np.logical_not(in_square),
                                         np.logical_not(in_square1)))
    
def isStateValid(state):
    # Some arbitrary condition on the state (note that thanks to
    # dynamic type checking we can just call getX() and do not need
    # to convert state to an SE2State.)
    #return state.getX() < .6
    x = state[0]
    y = state[1]
    return bool(obstacles(x,y))

goal = np.array([1.0,1.0])
def sample():
    if np.random.rand()<.8:
        return np.random.rand(2)*2-1
    else: #goal bias
        return goal

def collision_free(from_node,to_point):
    """
    check that the line in between is free
    """
    #print from_node,to_point,'collision_free'
    (x,y) = from_node['state']
    (x1,y1) = to_point
    #endpoints_free = obstacles(x,y) and obstacles(x1,y1)
    if not obstacles(x,y):
        return [],False
    direction = to_point-from_node['state']
    l = np.linalg.norm(direction)
    direction /= l
    step = .01
    
    free_points = []
    all_the_way = True
    for i in range(int(l/step)):
        (x,y) = direction*step*(i+1)+from_node['state']
        if( obstacles(x,y)):
            free_points.append(np.array([x,y]))
        else:
            all_the_way = False
            break
    
    #add the to_point if itś not in an obstacle and the line was free
    if all_the_way and obstacles(x1,y1):
        free_points.append(np.array(to_point))
    else:
        all_the_way = False
    return free_points, all_the_way

def steer(x_from,x_toward):
    extension_direction = x_toward-x_from
    norm = np.linalg.norm(extension_direction)
    if norm > 1:
        extension_direction = extension_direction/norm
    control = 1e-1 *extension_direction #steer
    
    x_new = x_from + control 
    return (x_new,control)
    
def distance(from_node,to_point):
    #to_point is an array and from_point is a node
    return np.linalg.norm(to_point-from_node['state'])
    
            
start = np.array([-1,-1])*1    
rrt = RRT(state_ndim=2)

rrt.set_distance(distance)
rrt.set_steer(steer)
rrt.set_goal_test(lambda state: False )
rrt.set_sample(sample)
rrt.set_collision_check(isStateValid)
rrt.set_collision_free(collision_free)

rrt.set_start(start)
rrt.init_search()
rrt.search(iters=2e3)

ax = plt.figure(None).add_subplot(111)

tree = rrt.tree
nx.draw_networkx(G=tree,
                 pos=nx.get_node_attributes(tree,'state'),
                 ax=ax,
                 node_size=25,
                 node_color=nx.get_node_attributes(tree,'cost').values(),
                 cmap = mpl.cm.get_cmap(name='copper'),
                 edge_color=nx.get_edge_attributes(tree,'pruned').values(),
                with_labels=False,
                #style='dotted'
                )
              
x = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,x)
o = obstacles(X,Y) #rasterize the obstacles
ax.imshow(o,origin='lower',extent=[-1,1,-1,1],alpha=.5)    

xpath = np.array([rrt.tree.node[i]['state'] for i in rrt.best_solution(goal)]).T
ax.plot(xpath[0],xpath[1],'g--',lw=10,alpha=.7)

plt.show()
