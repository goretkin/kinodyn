# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:17:45 2012

@author: gustavo
"""

import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt

tree = nx.DiGraph()

start = np.array([-1,-1])*1

node_id = 0
tree.add_node(node_id,attr_dict={'state':start,'hops':0,'cost':0})

dimension = 2
node_id +=1 

#return true for if point is not in collision
def obstacles(x,y):
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
    
def distance(from_node,to_point):
    #to_point is an array and from_point is a node
    return np.linalg.norm(to_point-from_node['state'])
    
def nearest_neighbor(graph,point):
    node_so_far = None
    distance_so_far = None
    for this_node in tree.nodes_iter():
        this_distance = distance(graph.node[this_node],point) 
        if(distance_so_far is None):            
            node_so_far = this_node
            distance_so_far = this_distance 
        elif(distance_so_far > this_distance):
            node_so_far = this_node
            distance_so_far = this_distance
        
    return (node_so_far,distance_so_far)

def near(graph,point,radius):
    S = {}    
    for this_node in tree.nodes_iter():
        this_distance = distance(graph.node[this_node],point)
        if(this_distance<radius):
            S[this_node] = this_distance
    return S
    
def k_nearest_neighbor(graph,point,k):
    ###return list of nodes sorted by distance from point
    H =[]
    heapsize = 0
    for this_node in tree.nodes_iter():
        this_distance = distance(graph.node[this_node],point)
        if(heapsize<k):
            heapq.heappush(H,(this_distance,this_node))
        else:
            heapq.heappushpop(H,(this_distance,this_node))
    S = [None]*k
    for i in range(k):
        S[k-i-1]=heapq.heappop(H)[1] #extract node ID
    return S
    
def sample():
    return np.random.rand(2)*2-1

def collision_free(from_node,to_point):
    #print from_node,to_point,'collision_free'
    (x,y) = from_node['state']
    (x1,y1) = to_point
    endpoints_free = obstacles(x,y) and obstacles(x1,y1)
    if not endpoints_free:
        return False
    direction = to_point-from_node['state']
    l = np.linalg.norm(direction)
    direction /= l
    step = .01
    for i in range(int(l/step)):
        (x,y) = direction*step*i+from_node['state']
        if( not obstacles(x,y)):
            return False
    return True
    
gamma_rrt = 1
eta = .5
c = 1
for i in range(500):
    x_rand = sample()*1
    x_nearest_id, _a  = nearest_neighbor(tree,x_rand)
    x_nearest = tree.node[x_nearest_id]['state']
    extension_direction = x_rand - x_nearest
    extension_direction = extension_direction/np.linalg.norm(extension_direction)
    x_new = x_nearest+.5*extension_direction #steer
    
    if(collision_free(tree.node[x_nearest_id],x_new)):
    
        cardinality = len(tree.node)
        radius = gamma_rrt * (np.log(cardinality)/cardinality)**(1.0/dimension)
        radius = np.min((radius,eta))
        print i,radius
        X_near = near(tree,x_new,radius)        
                
        x_min = x_nearest_id
        c_min = tree.node[x_min]['cost'] + c*distance(tree.node[x_min],x_new)
        
        #connect x_new to lowest-cost parent
        for x_near in X_near:
            this_cost = tree.node[x_near]['cost'] + c*distance(tree.node[x_near],x_new) 
            
            if collision_free(tree.node[x_near],x_new) and this_cost < c_min:
                x_min = x_near
                c_min = this_cost
        
        x_new_id = node_id
        node_id += 1
        tree.add_node(x_new_id,attr_dict={'state':x_new,
                                         'hops':1+tree.node[x_min]['hops'],
                                         'cost':tree.node[x_min]['cost']+distance(tree.node[x_min],x_new)
                                         }
                                         )
        tree.add_edge(x_min,x_new_id,attr_dict={'pruned':0})
                
        X_near = [] #don't rewire
        #rewire        
        for x_near in X_near:
            this_cost = tree.node[x_new_id]['cost'] + c*distance(tree.node[x_near],x_new) 
            
            if (collision_free(tree.node[x_new_id],tree.node[x_near]['state'])
                and this_cost < tree.node[x_near]['cost']):
                #better parent exists
                old_parent = tree.predecessors(x_near)
                if(True):  #don't keep pruned edges
                    assert len(old_parent)==1 #tree -- only one parent
                    old_parent = old_parent[0]
                    tree.remove_edge(old_parent,x_near)
                else:
                    true_parent = []
                    for parent in old_parent:
                        if tree.edge[parent][x_near]['pruned']==0:
                            true_parent.append(parent)
                    assert len(true_parent)==1
                    old_parent = true_parent[0]
                    tree.add_edge(old_parent,x_near,attr_dict={'pruned':1})
                
                tree.add_edge(x_new_id,x_near,attr_dict={'pruned':0})


#s = near(tree,np.array([.5,.5]),.1)
#s = k_nearest_neighbor(tree,np.array([.5,.5]),100)
#for node in s:
#    tree.node[node]['hops']= -1


nx.draw_networkx(tree,pos=nx.get_node_attributes(tree,'state'),
                 node_size=50,
                 node_color=nx.get_node_attributes(tree,'cost').values(),
                 edge_color=nx.get_edge_attributes(tree,'pruned').values(),
                with_labels=False,
                #style='dotted'
                )

x = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,x)
o = obstacles(X,Y)
plt.imshow(o,origin='lower',extent=[-1,1,-1,1],alpha=.5)    
            
plt.show()            
    
            
        