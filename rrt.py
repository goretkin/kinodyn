# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:17:45 2012

@author: gustavo
"""

import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
import matplotlib as mpl

def distance(from_node,to_point):
    #to_point is an array and from_point is a node
    return np.linalg.norm(to_point-from_node['state'])
    
class RRT():
    def __init__(self,state_ndim,keep_pruned_edges=False):
        self.tree = tree = nx.DiGraph()
        self.state_ndim = state_ndim
        self.next_node_id = 0
        
        self.gamma_rrt = 1
        self.eta = .5
        self.c = 1
        
        self.n_pruned = 0
        self.keep_pruned_edges = keep_pruned_edges
        
    def get_node_id(self):
        _id = self.next_node_id
        self.next_node_id += 1
        return _id

    def set_start(self,state0):
        assert len(state0) == self.state_ndim
        self.state0 = np.array(state0).reshape((self.state_ndim,))
    
    def set_goal_test(self,goal_test):
        """
        goal_test(state) = True/False
        """
        self.goal_test = goal_test
        
    def set_collision_check(self,collision_check):
        """
        collision_test(state) = True/False
        """
        self.collision_check = collision_check
        
    def set_sample(self,sample):
        """
        sample() returns point in state space
        """
        self.sample = sample
    
    def set_steer(self,steer):
        """
        steer(start,toward) returns a tuple (xnew, u)
        where xnew is point in the direction of toward from start
        and where u is the control action to apply
        """
        self.steer = steer

    def near(self,point,radius):
        """
        return a dictionary where keys are nodes and values are distances
        """
        S = {}    
        for this_node in self.tree.nodes_iter():
            this_distance = distance(self.tree.node[this_node],point)
            if(this_distance<radius):
                S[this_node] = this_distance
        return S
    
    def nearest_neighbor(self,state):
        node_so_far = None
        distance_so_far = None
        for this_node in self.tree.nodes_iter():
            this_distance = distance(self.tree.node[this_node],state) 
            if(distance_so_far is None):            
                node_so_far = this_node
                distance_so_far = this_distance 
            elif(distance_so_far > this_distance):
                node_so_far = this_node
                distance_so_far = this_distance
        return (node_so_far,distance_so_far)
        
    def k_nearest_neighbor(self,point,k):
        ###return list of nodes sorted by distance from point
        H =[]
        heapsize = 0
        for this_node in self.tree.nodes_iter():
            this_distance = distance(self.tree.node[this_node],point)
            if(heapsize<k):
                heapq.heappush(H,(this_distance,this_node))
            else:
                heapq.heappushpop(H,(this_distance,this_node))
        S = [None]*k
        for i in range(k):
            S[k-i-1]=heapq.heappop(H)[1] #extract node ID
        return S
    
    def search(self,iters=5e2):
        self.waste = 0

        c=1 #cost
        
        tree = self.tree
        tree.add_node(self.get_node_id(),
                      attr_dict={'state':self.state0,'hops':0,'cost':0})

        for i in xrange(iters):
            x_rand = self.sample()
            x_nearest_id, _a  = self.nearest_neighbor(x_rand)
            x_nearest = tree.node[x_nearest_id]['state']
            (x_new, _a) = self.steer(x_nearest,x_rand)
            
            #determine who the parent of x_new should be            
            free_points, all_the_way = collision_free(tree.node[x_nearest_id],x_new)
            
            if len(free_points) == 0:
                break
            
            if not all_the_way:
                x_new = free_points[-1]
                print 'not all the way'
            else:
                if not np.linalg.norm(np.array(free_points[-1]) - x_new) < 1e-5:
                    print np.linalg.norm(np.array(free_points[-1]) - x_new)
                    raise AssertionError()

            cardinality = len(tree.node)
            radius = self.gamma_rrt * (np.log(cardinality)/cardinality)**(1.0/self.state_ndim)
            radius = np.min((radius,self.eta))
            print i,self.n_pruned,self.waste
            
            X_near = self.near(x_new,radius)        
                    
            x_min = x_nearest_id
            c_min = tree.node[x_min]['cost'] + c*distance(tree.node[x_min],x_new)
            
            #connect x_new to lowest-cost parent
            for x_near in X_near:
                this_cost = tree.node[x_near]['cost'] + c*distance(tree.node[x_near],x_new) 
                
                #cheaper to check first condition
                if this_cost < c_min and collision_free(tree.node[x_near],x_new)[1]:
                    x_min = x_near
                    c_min = this_cost
            
            x_new_id = self.get_node_id()
            tree.add_node(x_new_id,attr_dict={'state':x_new,
                                             'hops':1+tree.node[x_min]['hops'],
                                             'cost':tree.node[x_min]['cost']+distance(tree.node[x_min],x_new)
                                             }
                                             )
            tree.add_edge(x_min,x_new_id,attr_dict={'pruned':0})
                    
            #X_near = [] #don't rewire
            discard_pruned_edge = not self.keep_pruned_edges
            #rewire to see if it's cheaper to go through the new point
            for x_near in X_near:
                this_cost = tree.node[x_new_id]['cost'] + c*distance(tree.node[x_near],x_new) 
                
                if (this_cost < tree.node[x_near]['cost'] and
                    collision_free(tree.node[x_new_id],tree.node[x_near]['state'])[1]
                    ):
                    #better parent exists
                    old_parent = tree.predecessors(x_near)
                    if(discard_pruned_edge):  #don't keep pruned edges
                        assert len(old_parent)==1 #each node in tree has only one parent
                        old_parent = old_parent[0]
                        tree.remove_edge(old_parent,x_near)
                    else:
                        #we are keeping edges that a pruned, so a node might 
                        #actually have more than one parent
                        true_parent = []
                        for parent in old_parent:
                            if tree.edge[parent][x_near]['pruned']==0:
                                true_parent.append(parent)
                        assert len(true_parent)==1
                        old_parent = true_parent[0]
                        tree.add_edge(old_parent,x_near,attr_dict={'pruned':1})
                    
                    tree.add_edge(x_new_id,x_near,attr_dict={'pruned':0})
                    
                    self.n_pruned += 1
    def best_solution(self,x):
        """
        return list of node IDs forming a path
        starting at the root node that comes closest to x
        """
        assert len(x)==self.state_ndim

        path = []        
        parent = self.nearest_neighbor(x)[0]
        path.append(parent)
        
        while(parent != 0): #until you get at the root
            possible_parents = self.tree.predecessors(parent)
            if(self.keep_pruned_edges):
                parent = filter(lambda node_id: self.tree.node[node_id]['pruned']==0,
                                possible_parents)
            else:
                parent = possible_parents
                
            assert len(parent)==1
            parent = parent[0]
            path.append(parent)

            
        return path[::-1] #reverse
                           
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
    if np.random.rand()<.9:
        return np.random.rand(2)*2-1
    else: #goal bias
        return goal

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
    
    free_points = []
    all_the_way = True
    for i in range(int(l/step)):
        (x,y) = direction*step*(i+1)+from_node['state']
        if( obstacles(x,y)):
            free_points.append(np.array([x,y]))
        else:
            all_the_way = False
            break
    free_points.append(np.array(to_point))
    return free_points, all_the_way

def steer(x_from,x_toward):
    extension_direction = x_toward-x_from
    norm = np.linalg.norm(extension_direction)
    if norm > 1:
        extension_direction = extension_direction/norm
    control = 1e-1 *extension_direction #steer
    
    x_new = x_from + control 
    return (x_new,control)
            
start = np.array([-1,-1])*1    
rrt = RRT(state_ndim=2)
rrt.set_steer(steer)
rrt.set_goal_test(lambda state: False )
rrt.set_sample(sample)
rrt.set_collision_check(isStateValid)
rrt.set_start(start)
rrt.search(iters=2e3)

ax = plt.figure(None).add_subplot(111)

tree = rrt.tree
nx.draw_networkx(G=tree,
                 pos=nx.get_node_attributes(tree,'state'),
                 ax=ax,
                 node_size=50,
                 node_color=nx.get_node_attributes(tree,'cost').values(),
                 cmap = mpl.cm.get_cmap(name='copper'),
                 edge_color=nx.get_edge_attributes(tree,'pruned').values(),
                with_labels=False,
                #style='dotted'
                )
                
xpath = np.array([rrt.tree.node[i]['state'] for i in rrt.best_solution(goal)]).T



x = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,x)
o = obstacles(X,Y) #rasterize the obstacles
ax.imshow(o,origin='lower',extent=[-1,1,-1,1],alpha=.5)    

ax.plot(xpath[0],xpath[1],'g--',lw=3,alpha=.7)            
plt.show()
    
            
        