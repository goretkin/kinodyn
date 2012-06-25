# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:55:29 2012

@author: gustavo
"""

import examplets
from rrt import RRT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    if norm > .5:
        extension_direction = extension_direction/norm
        extension_direction *= .5
    control = extension_direction #steer
    
    x_new = x_from + control 
    return (x_new,control)
    
def distance(from_node,to_point):
    #to_point is an array and from_point is a node
    return np.linalg.norm(to_point-from_node['state'])
            
start = np.array([-1,-1])*1    
rrt = RRT(state_ndim=2,keep_pruned_edges=False)

rrt.set_distance(distance)
rrt.set_steer(steer)
rrt.set_goal_test(lambda state: False )
rrt.set_sample(sample)
rrt.set_collision_check(isStateValid)
rrt.set_collision_free(collision_free)

rrt.gamma_rrt = 10.0
rrt.eta = 1.0
rrt.c = 1

rrt.set_start(start)
rrt.init_search()

import copy
ani_rrt = copy.deepcopy(rrt)

x = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,x)
obstacle_bitmap = obstacles(X,Y) #rasterize the obstacles

if False:
    rrt.search(iters=1e3)
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
                  
    ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5)    

    xpath = np.array([rrt.tree.node[i]['state'] for i in rrt.best_solution(goal)]).T
    ax.plot(xpath[0],xpath[1],'g--',lw=10,alpha=.7)

    plt.show()
    assert False


ani_fig = plt.figure(None)
ani_ax = ani_fig.gca()

ani_ax.set_xlim(-1,1)
ani_ax.set_ylim(-1,1)
ani_ax.set_aspect('equal')

ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5)    

#import copy


# shift the axis to make room for legend
box = ani_ax.get_position()
ani_ax.set_position([box.x0-.1, box.y0, box.width, box.height])


def update_frame(i): 
    print 'frame: ',i 
    ani_ax.cla()
    ani_ax.set_xlim(-1,1)
    ani_ax.set_ylim(-1,1)
    ani_ax.set_aspect('equal')    
    ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5,zorder=1)    
    
    ani_rrt.force_iteration()
    ani_ax.set_title('time index: %d'%(i))
    
    for l in ani_ax.lines:
        l.remove()
    for p in ani_ax.patches:
        p.remove()
        
    tree = ani_rrt.tree

    viz_x_nearest = tree.node[ani_rrt.viz_x_nearest_id]['state']
    viz_x_new = tree.node[ani_rrt.viz_x_new_id]['state']
    viz_x_from = tree.node[ani_rrt.viz_x_from_id]['state']    
    
    xpath = np.array([tree.node[i]['state'] for i in ani_rrt.best_solution(goal)]).T
    ani_ax.plot(xpath[0],xpath[1],ls='--',lw=10,alpha=.7,color=(.2,.2,.2,1),zorder=2,label='best path so far')

    ani_ax.plot([viz_x_from[0],viz_x_new[0]],[viz_x_from[1],viz_x_new[1]],'y',lw=5,alpha=.7,zorder=3,label='new extension')

    pos=nx.get_node_attributes(tree,'state')

    node_collection = nx.draw_networkx_nodes(G=tree,
                                            pos=pos,
                                            ax=ani_ax,
                                            node_size=25,
                                            node_color=nx.get_node_attributes(tree,'cost').values(),
                                            cmap = mpl.cm.get_cmap(name='copper'),
                                            )
    
    edge_collection = nx.draw_networkx_edges(G=tree,
                                            pos=pos,
                                            ax=ani_ax,
                                            edge_color=nx.get_edge_attributes(tree,'pruned').values(),
                                            )
    edge_collection.set_zorder(4)
    node_collection.set_zorder(5)
    
    
    #mfc is marker face color

    ani_ax.add_patch(mpl.patches.Circle(xy=viz_x_new,radius=ani_rrt.viz_search_radius,
                                        alpha=.3,fc='none',ec='b',label='_rewire radius'))

    ani_ax.plot(*ani_rrt.viz_x_rand,marker='*', mfc='k', mec='k', ls='None', zorder=6, label='x_rand')
    ani_ax.plot(*viz_x_nearest,marker='p', mfc='c', mec='c', ls='None', zorder=7, ms=5, label='x_nearest')
    ani_ax.plot(*viz_x_new, marker='x', mfc='r', mec='r', ls='None', zorder=8, label='x_new')    
    ani_ax.plot(*viz_x_from, marker='o', mfc='g',mec='g', ls='None',alpha=.5, zorder=9, label='x_from')



    ani_ax.legend(bbox_to_anchor=(1.05,0.0),loc=3,
                   ncol=1, borderaxespad=0.,
                    fancybox=True, shadow=True,numpoints=1)

#    ani_ax.legend(ncol=2)
    plt.setp(ani_ax.get_legend().get_texts(),fontsize='small')


    
    
    
ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=1000,interval=500)
ani.save('rrt.mp4', fps=5, codec='mpeg4', clear_temp=False)
#ani.save('test.mp4', fps=20, codec='mpeg4', clear_temp=True)
    
    

