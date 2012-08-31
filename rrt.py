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
    def __init__(self,state_ndim,control_ndim):

        self.tree = tree = nx.DiGraph()
        self.state_ndim = state_ndim
        self.control_ndim = control_ndim

        self.next_node_id = 0
        
        self.gamma_rrt = 1.0 #decay rate of ball
        self.eta = 0.5  #maximum ball size
        
        self.steer_reach_threshold = 1e-4 #self.distance(from_node,to_state) < self.steer_reach_threshold, then the two states are taken to be equivalent

        self.extension_aggressiveness = 1    #how many nodes to try to extend from if the nearest does not yield an extension
        self.rrt_until_feasible = True      #do RRT until you can start pruning, then so RRT*
        self.search_initialized = False 
        

        self.n_pruned = 0   #nodes removed due to pruning
        self.n_rewired = 0  #edges removed due to rewiring
        self.n_extensions = 0 #number of times an extension was attempted
        self.n_iters = 0
        
        self.found_feasible_solution = False
        self.worst_cost = None          #an upper-bound on the cost of a feasible solution. gets set after the first feasible solution is found
        self.can_prune = False          #if True, then worst_cost has decreased since last time we did a prune.

        self.deleted_nodes = set()      #nodes can be deleted as a result of pruning or as a result of rewiring causing a collision.
                                        #this set gets emptied at the beginning of an extend, and reflects the nodes deleted within a single iteration

        self.cheapest_goal = None   #the goal corresponding with the upper-bound cost
        self.goal_set_nodes = set() #a set of node ids that are within the goal set

        self.cost_history = []
        self.sample_history = []
        #check that whenever a node's cost is updated, it's not increased.
        self.check_cost_decreasing = True
        
        #visualization
        self.viz_change = False #did any of these data members get updated 
        self.viz_x_rand = None  #sampled point

        self.viz_x_nearest_id = None #point nearest to x_rand
        self.viz_x_new_id = None #extend till        
        self.viz_x_from_id = None #point to extend from
        self.viz_x_near_id = None #list of nodes that are within the search radius
        
        self.viz_x_nearest = None
        self.viz_x_new = None
        self.viz_x_from  = None
        self.viz_x_near_id = None #list of states that are within the search radius
        
        self.viz_search_radius = None
        self.viz_collided_paths = [] #collision queries that return collision, set to None to not store this information


        self.save_vars = [
                            'tree','state_ndim','next_node_id','gamma_rrt','steer_reach_threshold','extension_aggressiveness','rrt_until_feasible','search_initialized',
                            'n_pruned', 'n_extensions', 'n_iters',
                            'found_feasible_solution', 'worst_cost', 'can_prune', 'deleted_nodes',
                            'cheapest_goal', 'goal_set_nodes', 'cost_history', 'sample_history', 'check_cost_decreasing',
                            ]

        self.debug = True
                          
    def save(self,shelf_file):
        try:
            self.check_consistency()
        except AssertionError as e:
            print "Warning! saving an inconsistent RRT! ",e
        
        for var in self.save_vars:
            shelf_file[var] = self.__dict__[var]
            
    def load(self,shelf_file):
        if not self.search_initialized:
            print "Warning! initializing after loading will over-write the loaded values."
        for var in self.save_vars:
            if var not in shelf_file.keys():
                raise AssertionError('shelf file is missing key: %s'%str(var))
            self.__dict__[var] = shelf_file[var]
        self.check_consistency()
            
    def get_node_id(self):
        _id = self.next_node_id
        self.next_node_id += 1
        return _id

    def set_start(self,state0):
        assert len(state0) == self.state_ndim
        self.state0 = np.array(state0).reshape((self.state_ndim,))
    
    def set_goal_test(self,goal_test):
        """
        goal_test(node) = True/False
        """
        self.goal_test = goal_test
    
    def set_distance_from_goal(self,distance_from_goal):
        """
        goal_distance(node) = distance
        
        needed since goal is typically a set, not a single point.
        this is used for pruning, and pruning will break if this distance is an overestimate. 
        """
        self.distance_from_goal = distance_from_goal
        
    def set_collision_check(self,collision_check):
        """
        collision_test(state) = True/False
        """
        self.collision_check = collision_check
        
    def set_collision_free(self,collision_free):
        """
        collision_test(node,action) = x_path, u_path, all_the_way
        """
        if self.viz_collided_paths is None:
            self.collision_free = collision_free  
        else:
            #wrap collision checker in a function that will store paths that collided
            import types
            def _collision_free(self,node,action):
                x_path, u_path, all_the_way = collision_free(node,action)
                if not all_the_way:
                    self.viz_collided_paths.append( (node,action) ) 
                return x_path,u_path,all_the_way

            self.collision_free = types.MethodType(_collision_free,self,RRT) #bind method to self. invoke like self.collision_free(..)
                            
        
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

    def set_cost(self,cost):
        # cost(x,action) x is a starting state and action is an action
        self.cost = cost

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
        k = min(k,len(self.tree.nodes())) #handles the case where k is greater than the number of nodes
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
        nop_action = np.zeros(shape=(0,self.control_ndim))
        self.tree.add_node(self.start_node_id,
          attr_dict={'state':self.state0,'action':nop_action,'hops':0,'cost':0})

    def force_iteration(self,quiet=True):
        """
        sometimes during an iteration of RRT, a sampled state is such that
        no extensions are feasible, and so no new points are added.
        this keeps sampling until there is a success
        """
        n = len(self.tree.node)
        while(True):
            if not quiet: print 'attempt force_iteration'
            self.search(iters=1)
            if len(self.tree.node)>n:
                return
            
    def search(self,iters=5e2):
        for i in xrange(int(iters)):
            self.n_iters += 1
            x_rand = self.sample()
            self.extend(x_rand)
            print 'iter:',i,' n_pruned:',self.n_pruned, 'n_rewired:', self.n_rewired, 'nodes in tree:', len(self.tree.node)

    def check_goal(self,node_id):
        x_new_id = node_id
        tree = self.tree
        
        if self.goal_test(tree.node[x_new_id]):
                print 'added point in the goal set'
                self.goal_set_nodes.add(x_new_id)
                if not self.found_feasible_solution:
                    print '!!!\n'*5
                    print 'found first solution'
                    print '!!!\n'*5
                    self.found_feasible_solution = True
                    self.worst_cost = tree.node[x_new_id]['cost']
                    self.cheapest_goal = x_new_id
                    self.cost_history.append((self.n_iters,self.worst_cost))
                    self.can_prune = True
                else:
                    if tree.node[x_new_id]['cost']<self.worst_cost:         #there's a node in the goal that has a lowers the maximum cost (therefore we can prune more aggressively
                        self.worst_cost = tree.node[x_new_id]['cost']
                        self.cheapest_goal = x_new_id
                        self.cost_history.append((self.n_iters,self.worst_cost))
                        self.can_prune = True

    def prune(self):
        pruned_nodes = set()
        if self.do_pruning and self.can_prune:
            print 'Prune the tree: ',self.worst_cost
            pruned_nodes = self.prune_at_bound(self.worst_cost)
            print ' removed %d nodes'%len(pruned_nodes)
            self.n_pruned += len(pruned_nodes)
            if self.cheapest_goal in pruned_nodes:
                raise AssertionError("Pruning removed the best goal, which is used to set the pruning cost bound.")
        self.deleted_nodes = self.deleted_nodes.union(pruned_nodes)
        return pruned_nodes
            
    def extend_from(self,node_id,to_state):
        tree = self.tree

        x_actual,action = self.steer(tree.node[node_id],to_state)
        x_path, u_path, all_the_way  = self.collision_free(tree.node[node_id], action)

        new_id = self.get_node_id()
        if len(x_path) == 0:
            return None

        tree.add_node(new_id,attr_dict={  'state': x_path[-1],
                                            'action':u_path,
                                            'hops':1+tree.node[node_id]['hops'],
                                            'cost':tree.node[node_id]['cost']+self.cost(tree.node[node_id]['state'],u_path)
                                         }
        )
        tree.add_edge(node_id,new_id)
        return new_id

    
    def extend(self,x_rand):
        self.n_extensions += 1
        self.sample_history.append(x_rand)
        self.deleted_nodes = set()
        #this is what gives RRT* optimality. Set to False for vanilla RRT.                        
        do_find_cheapest_parent = self.found_feasible_solution or not self.rrt_until_feasible #do RRT until a solution is found, then proceed as RRT*

        #doesn't give optimality, but speeds up convergence.
        do_rewire = self.found_feasible_solution or not self.rrt_until_feasible
        
        #when adding an extension, add intermediate points
        add_intermediate_nodes = True    

        self.do_pruning = True

        extension_attempts = 1 #number of attempts of aggressive extension

        tree = self.tree
        cardinality = len(tree.node)
        radius = self.gamma_rrt * (np.log(cardinality)/cardinality)**(1.0/self.state_ndim)
        radius = np.min((radius,self.eta))          #radius of search ball

        x_nearest_id, nearest_distance  = self.nearest_neighbor(x_rand)
        if nearest_distance == np.inf:
            if self.debug: print 'no nearest node'
            return      #there is no nearest node (occurs for some distance functions)
        (x_new, action) = self.steer(tree.node[x_nearest_id],x_rand)

        #action drives from x_nearest toward x_rand, and actually lands at x_new
        if self.extension_aggressiveness == 'auto':
            extension_aggressiveness = max(1,len(tree.nodes())/10) #the number of nodes to try extension from.
        else:
            extension_aggressiveness = self.extension_aggressiveness

        x_path, u_path, all_the_way = self.collision_free(tree.node[x_nearest_id],action)

        if len(x_path) == 0:
            """
            not possible to extend the x_nearest
            """
            if extension_aggressiveness <=1:
                if self.debug: print 'no collision-free extension possible'
                return #go to next iteration
            else:
                #candidate_x_nearest is thusly denoated "candidate" because extension from it may not be possible.
                x_path = None #trying to debug
                all_the_way = None
                x_nearest_id = None
                action = None
                for candidate_x_nearest_id in self.k_nearest_neighbor(x_rand,extension_aggressiveness)[1:]:
                    extension_attempts += 1
                    (candidate_x_new, action) = self.steer(tree.node[candidate_x_nearest_id],x_new)
                    (x_path, u_path, all_the_way) = self.collision_free(tree.node[candidate_x_nearest_id],action)
                    if len(x_path) > 0:
                        x_nearest_id = candidate_x_nearest_id   
                        x_new = candidate_x_new     #Driving from a different parent to x_new cannot guarantee you land at x_new.
                        break

                if len(x_path) == 0:
                    if self.debug: print 'aggresive extension %d still found nothing to extend from!'%(extension_aggressiveness)
                    return
        
        if not all_the_way:
            x_new = x_path[-1]
        else:
            if not np.allclose(x_path[-1],x_new,atol=1e-4): #fixme bring out this parameter
                print 'error',np.linalg.norm(np.array(x_path[-1]) - x_new)
                print 'expected x_new',x_new
                print 'actual x_new',x_path[-1]
                print '\n\n\n\nraise SoftAssertion!!!!!\n\n\n\n'
                #raise AssertionError('steer function or collision_free function is inaccurate')    #fixme
            

        #by this point, we have an x_new that is collision-free reachable from at least one node in the tree (namely x_nearest_id)
        #determine who the parent of x_new should be

        #keep track of information pertinent to the best parent so far. Initialize with x_nearest_id        
        x_min = x_nearest_id       
        candidate_x_new_min = x_new         #seems strange that we need to store candidate_x_new_min, but in general steering will not exactly reach x_new, so different parents might have different x_new
        candidate_action_min = action       #action that drives x_min toward candidate_x_new, but might collide
        x_path_min = x_path                 #state trajectory from x_min to candidate_x_new
        u_path_min = u_path                 #control trajectory from x_min to candidate_x_new
#        c_min = tree.node[x_min]['cost']  + sum([self.cost(x,u) for (x,u) in zip([tree.node[x_nearest_id]['state']]+list(x_path_min[1:]),u_path_min)])      #cost of candidate_x_new_min. 
                                                                                                #can't simply do self.cost(tree.node[x_nearest_id]['state'],action) because action might cause a collision
        c_min = tree.node[x_min]['cost'] + self.cost(tree.node[x_nearest_id]['state'],u_path_min)

        if do_find_cheapest_parent or do_rewire:
            X_near = self.near(x_new,radius)
            print 'nodes in ball:',len(X_near)
        else:
            X_near = None
    
        if do_find_cheapest_parent:        
            #consider all nodes in X_near as potential parents for x_new
            #connect x_new to lowest-cost parent
            for x_near in X_near:                
                #cheaper to check first condition
                candidate_x_new, candidate_action = self.steer(tree.node[x_near],x_new)
                x_path, u_path, all_the_way = self.collision_free(tree.node[x_near],candidate_action) #would be great if didn't need to perform this step in order to compute the cost.

                if all_the_way and np.allclose(candidate_x_new,x_new):  #fixme allclose should have some greater tolerance for systems for which it is hard to steer
                    #this_cost = tree.node[x_near]['cost'] + sum([self.cost(x,u) for (x,u) in zip([tree.node[x_near]['state']]+list(x_path[1:]),u_path)])
                    this_cost = tree.node[x_near]['cost'] + self.cost(tree.node[x_near]['state'],u_path)
                    if this_cost < c_min:                
                        x_min = x_near
                        c_min = this_cost
                        candidate_x_new_min = candidate_x_new
                        candidate_action_min = candidate_action
                        x_path_min = x_path
                        u_path_min = u_path

        #the procedure above deemed action the best control to make from x_min in order to get to x_new
        action = candidate_action_min
        x_new = candidate_x_new_min

        if not add_intermediate_nodes:
            assert len(x_new) == self.state_ndim
            x_new_id = self.get_node_id()

            tree.add_node(x_new_id,attr_dict={  'state':x_new,
                                                'action':action,
                                                'hops':1+tree.node[x_min]['hops'],
                                                'cost':tree.node[x_min]['cost']+self.cost(tree.node[x_min]['state'],action)
                                             }
            )
            tree.add_edge(x_min,x_new_id)

        else:
            #segment the extension into tiny parts, as given by the collision_free function
            last_node_id = x_min
            for i in range(len(x_path_min)):
                x = x_path_min[i]
                u = u_path_min[[i]] #preserve the dimensionality
                assert len(x_new) == self.state_ndim 
                this_node_id = self.get_node_id()
                tree.add_node(this_node_id,attr_dict={  'state':x,
                                                        'action':u,
                                                        'hops':1+tree.node[last_node_id ]['hops'],
                                                        'cost':tree.node[last_node_id ]['cost']+self.cost(tree.node[last_node_id]['state'],u)
                                                     }
                )
                tree.add_edge(last_node_id,this_node_id)
                last_node_id = this_node_id                
            x_new_id = last_node_id

        #now x_new_id has a parent and is in the tree

        #another goal bias -- try to grow toward the goal.
        if not self.goal is None:
            added_id = self.extend_from(x_new_id,self.goal)
            if not added_id is None: 
                print 'goal extension bias got somewhere.'#,tree.node[added_id]['action']
                self.check_goal(added_id)

        self.viz_x_rand = x_rand
        self.viz_x_nearest_id = x_nearest_id            
        self.viz_x_new_id = x_new_id            
        self.viz_x_from_id = x_min

        #pruning might remove these nodes so store the visualization information
        self.viz_x_nearest = tree.node[x_nearest_id]['state']
        self.viz_x_new = tree.node[x_new_id]['state']
        self.viz_x_from = tree.node[x_min]['state']
        
        self.viz_search_radius = radius

        if X_near is not None:
            self.viz_x_near_id = X_near
            self.viz_x_near = [tree.node[i]['state'] for i in X_near]
        else:
            self.viz_x_near_id = None
            self.viz_x_near = None


        self.viz_change = True
        
        self.check_goal(x_new_id)
        pruned_nodes = self.prune()
        if do_rewire:
            if x_new_id in pruned_nodes:
                #pruning removed x_new
                print 'pruning removed x_new'
                return
            #if nodes were pruned, then X_near may contain invalid nodes
            #can re-do the query, or just remove
            X_near = set(X_near) - pruned_nodes 
        
            #rewire to see if it's cheaper to go through the new point x_new
            for x_near in X_near:
                if x_near in self.deleted_nodes: 
                    print 'updating dynamics removed something in X_near',x_near
                    continue
                #proposed_cost = tree.node[x_new_id]['cost'] + c*self.distance(tree.node[x_near],x_new)
                candidate_x_near, action = self.steer(tree.node[x_new_id],tree.node[x_near]['state'])   #can't steer exactly to x_near
                proposed_cost = tree.node[x_new_id]['cost'] + self.cost(tree.node[x_new_id]['state'],action)
                if (proposed_cost < tree.node[x_near]['cost']):
                    x_path, u_path, all_the_way =self.collision_free(tree.node[x_new_id],action)
                    if all_the_way:
                        if self.distance(tree.node[x_near],candidate_x_near) < self.steer_reach_threshold:
                            #rewire. parent of x_near should be x_new
                            print ' updating x_near from ', tree.node[x_near]['state'], ' to ' , candidate_x_near

                            old_parent = tree.predecessors(x_near)
                            assert len(old_parent)==1 #each node in tree has only one parent
                            old_parent = old_parent[0]
                            tree.remove_edge(old_parent,x_near)
                            tree.node[x_near]['state'] = candidate_x_near
                            tree.node[x_near]['action'] = action
                            tree.add_edge(x_new_id,x_near)

                            #x_near has a new parent, so in general we need to propagate the new cost and the new dynamics.
                            #enforce dynamics might wiggle the states around a little bit, changing the cost evaluation, so do that first.
                            self._deep_enforce_dynamics(x_near)
                            self._deep_update_cost(x_near,proposed_cost)

                    self.n_rewired += 1

        


    def _deep_enforce_dynamics(self,node_id):
        #node_id['state'] supposedly has moved. apply action of all the children 
        childs = list(self.tree.successors_iter(node_id))   #can't iterate over dictionary while removing entries
        for child in childs:
            x_path,u_path,all_the_way = self.collision_free(self.tree.node[node_id],self.tree.node[child]['action'])
            if not all_the_way:
                print 'updating dynamics removed tree rooted at ',child
                self.remove_subtree(child)
            else:
                state_orig = self.tree.node[child]['state']
                if len(x_path) == 0:
                    self.tree.node[child]['state'] = self.tree.node[node_id]['state']
                    print 'redundant node' #fixme
                else:    
                    self.tree.node[child]['state'] = x_path[-1]
                if not np.allclose(state_orig,self.tree.node[child]['state']):
                    print 'node ', child, 'wiggle from', state_orig, ' to ', self.tree.node[child]['state']
                self._deep_enforce_dynamics(child)

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
        
        self.tree.node[node_id]['cost'] = cost
        
        if node_id in self.goal_set_nodes:
            if cost<self.worst_cost:                
                #print "_deep_update_cost updated self.worst_cost from %f to %f"%(self.worst_cost,cost)
                self.worst_cost = cost
                self.cheapest_goal = node_id
                self.cost_history.append((self.n_iters,self.worst_cost))
        for child in tree.successors_iter(node_id):
            #new_cost = tree.node[node_id]['cost'] + self.distance(tree.node[node_id],tree.node[child]['state'])
            new_cost = tree.node[node_id]['cost'] + self.cost(tree.node[node_id]['state'],tree.node[child]['action'])
            #TODO cache distances
            #_new_cost = tree.node[node_id]['cost'] + (tree.node[child]['cost'] - tree.node[node_id]['cost'])
            #if not abs(new_cost - _new_cost)<1e-6:
            #    print '_new_cost %f new_cost %f'%(_new_cost,new_cost)
                
            self._deep_update_cost(child,new_cost)

    def get_action(self,node_ids):
        """
        pack actions into a single array of actions
        """
        upath = []
        for i in node_ids:             #the first action is None -- action at the root
            for control in self.tree.node[i]['action']:
                upath.append(control)
        upath = np.array(upath)
        
    def best_solution_goal(self):
        if self.cheapest_goal is None:
            return None
        graph = self.tree.reverse()
        #there's actually only a single path, since the graph is a tree
        path = nx.shortest_path(graph,source=self.cheapest_goal,target=self.start_node_id)
        path = path[::-1]
        upath = self.get_action(best_sol)   #the first node contains an empty action action

        return path, np.array([self.tree.node[i]['state'] for i in path]), upath.T

                
    def best_solution_(self,x):
        """
        return list of node IDs forming a path
        starting at the root node that comes closest to x
        """
        assert len(x)==self.state_ndim

        path = []        
        parent = self.nearest_neighbor(x)[0]
        path.append(parent)
        
        while(parent != self.start_node_id): #until you get at the root
            parent = self.tree.predecessors(parent)                
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
        #there's actually only a single path, since the graph is a tree
        path = nx.shortest_path(graph,source=goal_id,target=self.start_node_id)
        return path[::-1]
        
    #in order for prune to work, self.distance_from_goal cannot overestimate the true cost to get to the goal.
    def prune_at_bound(self,bound):
        nodes_before = set(self.tree.nodes())
        self._prune_from(bound,self.start_node_id)
        nodes_after = set(self.tree.nodes())
        assert nodes_after<=nodes_before #the new nodes should be a subset of the old nodes
        nodes_removed = nodes_before - nodes_after
        self.goal_set_nodes = self.goal_set_nodes - nodes_removed   #in case we removed goal
        return nodes_removed
        
    def _prune_from(self,bound,root):
        #prune the subtree rooted at root
        for this_id in self.tree.successors(root):
            best_possible_cost = self.tree.node[this_id]['cost'] + self.distance_from_goal(self.tree.node[this_id])
            if  best_possible_cost > bound:
                #print 'removing %d with best-case cost of %f'%(this_id,best_possible_cost)
                self.remove_subtree(this_id)
                def node_action(node):
                    node['cost']=10
                #self.do_to_subtree(this_id,node_action)
            else:
                self._prune_from(bound,this_id)            
        
    def remove_subtree(self,root_id):
        succs = self.tree.successors(root_id)        
        for node in succs:
            self.tree.remove_edge(root_id,node)
            self.remove_subtree(node)
            
        self.tree.remove_node(root_id)
        self.deleted_nodes.add(root_id)     #keep track of deleted nodes
        return

    def do_to_subtree(self,root_id,node_action=None,edge_action=None):
        """
        call node_action(node) and edge_action(edge) on every node and edge of the
        subtree rooted at root_id
        """
        succs = self.tree.successors(root_id)        
        for node in succs:
            if not edge_action is None: edge_action(self.tree.edge[(root_id,node)])    
            self.do_to_subtree(node,node_action,edge_action)
            
        if not node_action is None: node_action(self.tree.node[root_id])
        return
        
    def check_cost_consistency(self):
        for edge in self.tree.edges_iter():
            a,b = edge
            dcost = self.tree.node[b]['cost']-self.tree.node[a]['cost']
            distance = self.distance(self.tree.node[a],self.tree.node[b]['state'])
            error = abs(dcost-distance)
            if error > 1e-6:
                raise AssertionError('consistency check: edge: %s, dcost: %f, distance: %f, error:%f'%(str(edge),dcost,distance,error))

    def check_tree_constraint(self):
        #assert each node has exactly one parent except for the root.
        for node in self.tree.nodes():
            parents = self.tree.predecessors(node)
            if not node == self.start_node_id:
                if not len(parents) == 1:
                    raise AssertionError("Node %s does not have exactly one parent."%(str(node)))
            else:
                if not len(parents) == 0:
                    raise AssertionError("The supposed root of the tree actually has %d parents."%(len(parents)))

    def check_consistency(self):
        #check validity of RRT class:
        self.check_tree_constraint()

        if not set(self.tree.nodes()) >= self.goal_set_nodes:
            raise AssertionError("There are things in goal_set_nodes that are not in the tree.")
        for goal in self.goal_set_nodes:
            if not self.goal_test(self.tree.node[goal]): 
                raise AssertionError("There is a node in goal_set_nodes that does not pass the goal test.")

        final_costs = [self.tree.node[s]['cost'] for s in self.goal_set_nodes]
        if not (len(final_costs)>0) == self.found_feasible_solution:
            raise AssertionError("There isn't actually a feasible solution.")
        if(self.found_feasible_solution):
            if not abs(min(final_costs) - self.worst_cost) < 1e-10:
                raise AssertionError("Inconsistent worst_cost")

        self.check_cost_consistency()
    
        
