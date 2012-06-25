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
    
class RRT():
    def __init__(self,state_ndim,keep_pruned_edges=False):
        self.tree = tree = nx.DiGraph()
        self.state_ndim = state_ndim
        self.next_node_id = 0
        
        self.gamma_rrt = 1.0 #decay rate of ball
        self.eta = 0.5  #maximum ball size
        self.c = 1      #how the cost gets weighted
        
        self.search_initialized = False 
        
        self.n_pruned = 0
        self.keep_pruned_edges = keep_pruned_edges
        
        self.check_cost_decreasing = True
        
        #visualization
        self.viz_x_rand = None  #sampled point
        self.viz_x_nearest_id = None #point nearest to x_rand
        self.viz_x_new_id = None #extend till
        self.viz_x_from_id = None #point to extend from
        self.viz_search_radius = None
            
            
        
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
        
    def set_collision_free(self,collision_free):
        """
        collision_test(node,state) = [free states],all_the_way
        """
        self.collision_free = collision_free  
        
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
    def set_distance(self,distance):
        """
        distance(from_node,to_point)
        """
        self.distance = distance

    def near(self,point,radius):
        """
        return a dictionary where keys are nodes and values are distances
        """
        S = {}    
        for this_node in self.tree.nodes_iter():
            this_distance = self.distance(self.tree.node[this_node],point)
            if(this_distance<radius):
                S[this_node] = this_distance
        return S
    
    def nearest_neighbor(self,state):
        node_so_far = None
        distance_so_far = None
        for this_node in self.tree.nodes_iter():
            this_distance = self.distance(self.tree.node[this_node],state) 
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
            this_distance = self.distance(self.tree.node[this_node],point)
            if(heapsize<k):
                heapq.heappush(H,(this_distance,this_node))
            else:
                heapq.heappushpop(H,(this_distance,this_node))
        S = [None]*k
        for i in range(k):
            S[k-i-1]=heapq.heappop(H)[1] #extract node ID
        return S
    
    def init_search(self):
        """
        only call once
        """
        if(self.search_initialized):
            print "search already initializd"

        self.search_initialized = True
        
        self.start_node_id = self.get_node_id()
        self.tree.add_node(self.start_node_id,
          attr_dict={'state':self.state0,'hops':0,'cost':0})

    def force_iteration(self):
        """
        sometimes during an iteration of RRT, a sampled state is such that
        no extensions are feasible, and so no new points are added.
        this keeps sampled until there is a success
        """
        n = len(self.tree.node)
        while(True):
            self.search(iters=1)
            if len(self.tree.node)>n:
                return
            
    def search(self,iters=5e2):
        c=self.c #cost
        
        tree = self.tree

        for i in xrange(int(iters)):
            x_rand = self.sample()
            x_nearest_id, _a  = self.nearest_neighbor(x_rand)
            x_nearest = tree.node[x_nearest_id]['state']
            (x_new, _action) = self.steer(x_nearest,x_rand)
            
            #determine who the parent of x_new should be            
            free_points, all_the_way = self.collision_free(tree.node[x_nearest_id],x_new)
            
            if len(free_points) == 0:
                continue #go to next iteration
            
            if not all_the_way:
                x_new = free_points[-1]

            else:
                if not np.linalg.norm(np.array(free_points[-1]) - x_new) < 1e-5:
                    print np.linalg.norm(np.array(free_points[-1]) - x_new)
                    raise AssertionError()

            cardinality = len(tree.node)
            radius = self.gamma_rrt * (np.log(cardinality)/cardinality)**(1.0/self.state_ndim)
            radius = np.min((radius,self.eta))
            
            print 'iter:',i,' n_pruned:',self.n_pruned,' num free_points:',len(free_points)
            
            X_near = self.near(x_new,radius)        
                    
            x_min = x_nearest_id
            c_min = tree.node[x_min]['cost'] + c*self.distance(tree.node[x_min],x_new)
                        
            do_find_cheapest_parent = True

            if do_find_cheapest_parent:
                #connect x_new to lowest-cost parent
                for x_near in X_near:
                    this_cost = tree.node[x_near]['cost'] + c*self.distance(tree.node[x_near],x_new) 
                    
                    #cheaper to check first condition
                    if this_cost < c_min and self.collision_free(tree.node[x_near],x_new)[1]:
                        x_min = x_near
                        c_min = this_cost
            
            add_intermediate_nodes = True
            
            if not add_intermediate_nodes:
                x_new_id = self.get_node_id()    
                tree.add_node(x_new_id,attr_dict={'state':x_new,
                                                 'hops':1+tree.node[x_min]['hops'],
                                                 'cost':tree.node[x_min]['cost']+
                                                 self.distance(tree.node[x_min],x_new)
                                                 }
                                                 )
                tree.add_edge(x_min,x_new_id,attr_dict={'pruned':0})
            else:
                path,all_the_way = self.collision_free(tree.node[x_min],x_new)
                assert all_the_way #it was just true when we called
                
                last_node_id = x_min
                decimation_factor = 10
                decimated_path = path[:-1:decimation_factor ]
                decimated_path.append(path[-1]) #ensure final point is in the path, regardless of decimation
                
                for x in decimated_path:
                    this_node_id = self.get_node_id()
                    tree.add_node(this_node_id,attr_dict={'state':x,
                                                 'hops':1+tree.node[last_node_id ]['hops'],
                                                 'cost':tree.node[last_node_id ]['cost']+
                                                 self.distance(tree.node[last_node_id ],x)
                                                 }
                                                 )
                    tree.add_edge(last_node_id,this_node_id,attr_dict={'pruned':0})
                    last_node_id = this_node_id
                    
                x_new_id = last_node_id
            

            self.viz_x_rand = x_rand
            self.viz_x_nearest_id = x_nearest_id
            self.viz_x_new_id = x_new_id
            self.viz_x_from_id = x_min
            self.viz_search_radius = radius
            
            do_rewire = True
            if do_rewire:
                discard_pruned_edge = not self.keep_pruned_edges
                #rewire to see if it's cheaper to go through the new point
                for x_near in X_near:
                    #proposed_cost = tree.node[x_new_id]['cost'] + c*self.distance(tree.node[x_near],x_new)
                    proposed_cost = tree.node[x_new_id]['cost'] + c*self.distance(tree.node[x_new_id],tree.node[x_near]['state'])  
                    
                    if (proposed_cost < tree.node[x_near]['cost'] and
                        self.collision_free(tree.node[x_new_id],tree.node[x_near]['state'])[1]
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
                            ani_rrt.tree.edge[old_parent][x_near]['pruned']=1 #don't delete edge -- mark as pruned
                        
                        tree.add_edge(x_new_id,x_near,attr_dict={'pruned':0})
    
                        self._deep_update_cost(x_near,proposed_cost)

                        self.n_pruned += 1
        
    def _deep_update_cost(self,node_id,cost):
        """
        update the cost node_id and of all the children of node_id
        """
        tree = self.tree

        if self.check_cost_decreasing:
            if tree.node[node_id]['cost'] < cost:
                raise AssertionError('cost of node %d increased by %f'%(node_id,cost-tree.node[node_id]['cost']))
            else:
                print 'node %d decreased by %f'%(node_id,tree.node[node_id]['cost']-cost)
        
        for child in tree.successors_iter(node_id):
            new_cost = tree.node[node_id]['cost'] + self.distance(tree.node[node_id],tree.node[child]['state'])
            self._deep_update_cost(child,new_cost)
        
    def best_solution_(self,x):
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
    def best_solution(self,x):
        """
        return list of node IDs forming a path
        starting at the root node that comes closest to x
        """
        assert len(x)==self.state_ndim
        goal_id = self.nearest_neighbor(x)[0]
        graph = self.tree.reverse()
        #remove pruned edges if we kept them in the graph
        if self.keep_pruned_edges:            
            for e in graph.edges_iter():
                if graph.edge[e]['pruned']==1:
                    del graph.edge[e]
            
            
        #there's actually only a single path, since the graph is a tree
        path = nx.shortest_path(graph,source=goal_id,target=self.start_node_id)
        return path[::-1]          
        
