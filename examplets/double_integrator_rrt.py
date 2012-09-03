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

import itertools

from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR, final_value_LQR, lqr_dim

class LQR_RRT():
    def __init__(self,A,B,Q,R,max_time_horizon):
        (self.n,self.m) = lqr_dim(A,B,Q,R)
        (self.A,self.B,self.Q,self.R) = (A,B,Q,R)
        self.max_time_horizon = max_time_horizon
        
        def action_state_valid(x,u):
            return True

        self.action_state_valid = action_state_valid

    def run_forward(self,x0,us):
        A = self.A
        B = self.B
        n = self.n+1  #+1 for time dimension 
        m = self.m
        us = np.reshape(us,newshape=(-1,m))
        assert len(x0) == n
        T = us.shape[0]
        xs = np.zeros(shape=(T+1,n))
        xs[0] = np.squeeze(x0)

        for i in range(1,T+1):
            xs[i,0:self.n] = np.dot(A,xs[i-1,0:self.n].T) + np.dot(B,us[i-1].T)
            xs[i,self.n] = xs[i-1,self.n] + 1               #increase time
        return xs[1:]

    def run_forward_fb(self,x0,gain_schedule):
        A = self.A
        B = self.B
        n = self.n+1  #+1 for time dimension 
        m = self.m
        assert len(x0) == n
        T = gain_schedule.shape[0]
        assert gain_schedule.shape[2] == n-1 #no time in the gain 
        assert gain_schedule.shape[1] == m

        xs = np.zeros(shape=(T+1,n))
        us = np.zeros(shape=(T,m))

        xs[0] = np.squeeze(x0)

        for i in range(1,T+1):
            us[i-1] = -1* np.dot(gain_schedule[i-1],xs[i-1,0:self.n])
            xs[i,0:self.n] = np.dot(A,xs[i-1,0:self.n].T) + np.dot(B,us[i-1].T)
            xs[i,self.n] = xs[i-1,self.n] + 1
        return xs[1:],us


    def collision_free(self,from_node,action):
        """
        check that taking action from from_node produces a collision free trajectory
        if not, return a partial trajectory for the state (x_path) and control (u_path)
        u_path is a list of actions -- it partitions the actions.
        """

        x0 = from_node['state']
        x_path = []
        u_path = []
        all_the_way = True
        if len(action) > 0:
            if self.action_state_valid(x0,action[0]):
                x_path_np = self.run_forward(x0,action)
                for (x,u) in zip(x_path_np,action):
                    if not self.action_state_valid(x,u):
                        all_the_way = False
                        break
                    x_path.append(x)
                    u_path.append(u) 
        u_path = np.array(u_path)

        return x_path, u_path, all_the_way        

    def same_state(self,a,b):
        return a[self.n] == b[self.n] and np.allclose(a,b,atol=1e-4) #time has to be identical and phase-space state has to be approximate

    def cost(self,x_from,action):
        #this does not include the cost of being in at the state arrived by taking the last action
        Q = self.Q
        R = self.R
        assert len(x_from) == 3
        if len(action) == 0:
            #null action
            return 0    #is captured in dynamics below, but if the action is actually null, then the next assertion fails
        assert action.shape[1] == 1

        x_path = self.run_forward(x_from,action)
        cost = 0

        for i in range(action.shape[0]):
            #x_path does not include x_from
            x = x_path[[i-1],0:self.n].T if i>0 else x_from[0:self.n] #don't include time
            u = action[[i],:].T
            cost += np.squeeze( np.dot(u.T,np.dot(R,u))
                                +np.dot(x.T,np.dot(Q,x))
                                ) 
        return cost

    def node_cache_ctg(self,node):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        max_time_horizon = self.max_time_horizon

        #print 'calculate ctg for', node
        x = node['state']        
        #reverse system
        Ar = A.I
        Br = -A.I * B
        kmax = max_time_horizon - x[self.n] +1
        assert kmax > 0

        #ctg[0] is cost-to-go zero steps -- very sharp quadratic
        #so ctg[k] with k = max_time_horizon - from_node['state'][2] is time to go


        Fs, Ps = final_value_LQR(Ar,Br,Q,R,x[0:self.n],kmax)
        #storing in reverse order is easier to think about.
        #node['gain'][i] is what you should do with i steps left to go.
        node['ctg'] = Ps[::-1] 
        node['gain'] = Fs[::-1]

    def steer(self,x_from_node,x_toward):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R

        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<=0:
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))

        Qf = np.eye(self.n) * 1e8
        qf = -np.dot(Qf,desired)

        Qhf = np.zeros(shape=(self.n+1,self.n+1))
        Qhf[0:self.n,0:self.n] = Qf
        Qhf[0:self.n,[self.n]] = qf
        Qhf[[self.n],0:self.n] = qf.T
        Qhf[self.n,self.n] = np.dot(desired.T,np.dot(Qf,desired))

        (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                    B=B,
                                    c=np.zeros(shape=(self.n,1)),
                                    Q=Q,
                                    R=R,
                                    q=np.zeros(shape=(self.n,1)),
                                    r=np.zeros(self.m),
                                    ctdt='dt')
        #pk should be zeros. 

        Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)
        #print Fs
        xs = np.zeros(shape=(T+1,self.n+1))
        us = np.zeros(shape=(T,self.m))
        xs[0] = x_from

        for i in range(T):
            us[i] = -1 * (np.dot(Fs[i,:,0:self.n],xs[i,0:self.n]) + Fs[i,:,self.n])
            xs[i+1,0:self.n] = np.dot(A,xs[i,0:self.n].T) + np.dot(B,us[i].T)
            xs[i+1,self.n] = xs[i,self.n] + 1

        x_actual = xs[-1]    
        return (x_actual, us)


    def steer_cache(self,x_from_node,x_toward):
        A = self.A
        B = self.B

        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[2] - x_from[2] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T <= 0:
            return (x_from,np.zeros(shape=(0,1)))   #stay here

        if T < 10:
            #this technique isn't too accurate for short times due to slack in the final-value constraint
            #so do something else.
            return self.steer(x_from_node,x_toward)

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))            

        if 'gain' not in x_from_node:
            self.node_cache_ctg(x_from_node)

        Fs = x_from_node['gain']

        if T >len(Fs):
            print "requested uncached steer!!!"
            return self.steer(x_from_node,x_toward) #fixme should cache more

        #reverse system
        Ar = A.I
        Br = -A.I * B

        xs = np.zeros(shape=(T+1,self.n+1))
        us = np.zeros(shape=(T,1))

        #we're driving backwards. start at x_toward.
        xs[0] = x_toward

        for i in range(T):
            j = T - i-1 #gain matrices Fs[j] is what you should do with j steps remaining
            us[i] = -1 * (np.dot(Fs[j,:,0:self.n],xs[i,0:self.n]) + Fs[j,:,self.n])
            xs[i+1,0:self.n] = np.dot(Ar,xs[i,0:self.n].T) + np.dot(Br,us[i].T)
            xs[i+1,self.n] = xs[i,self.n] - 1 #reverse time

        xs = xs[::-1]
        us = us[::-1]
        x_actual = xs[-1]    
        #print 'error',(x_from - xs[0])
        return (x_actual, us)

    def steer_QP(self,x_from_node,x_toward):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R

        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<=0:
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here

        try:
            qpsol, qpmats ,xs,us = LQR_QP(A,B,Q,R,(T+1),
                                x0=x_from[0:self.n],
                                xT=x_toward[0:self.n])
        except ValueError as e:
            #quadratic program is probably infeasible. This can happen if the time horizon is too short and the system's reachability doesn't include the final-value constraint
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here (could do something smarter)
            
        (QP_P,QP_q,QP_A,QP_B) = qpmats

        xs = xs.T
        us = us.T
        x_actual = np.concatenate((     xs[-1],
                                        [x_from[self.n]+us.shape[0]]
                                ))
        return (x_actual, us)

    def distance_direct(self,from_node,to_point):
        #print from_node['state'], to_point
        #to_point is an array and from_point is a node
        assert len(to_point)==self.n+1
        x_actual,action = self.steer(from_node,to_point)
        if self.same_state(x_actual,to_point): #if actually drove there:
            return self.cost(from_node['state'],action)
        else:
            return np.inf

    def distance_direct_qp(self,from_node,to_point):
        #print from_node['state'], to_point
        #to_point is an array and from_point is a node
        assert len(to_point)==self.n
        x_actual,action = self.steer_QP(from_node,to_point)
        if self.same_state(x_actual,to_point): #if actually drove there:
            return self.cost(from_node['state'],action)
        else:
            return np.inf

    def distance(self,from_node,to_point):
        #to_point is an array and from_point is a node

        global A
        global B
        global Q
        global R

        x_from = from_node['state']
        x_toward = to_point
        assert len(x_toward)==self.n+1

        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<0:
            return np.inf
        elif T==0:
            return 0 if self.same_state(x_from,x_toward) else np.inf

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))            

        #we want the final bowl to be centered at desired:
        #(x-x_d)^T * Qf * (x-x_d)
        #xT*Qf*x -x_dT * Qf * x - xT *Qf *x_d * x_dT * Qf * x_d
        Qf = np.eye(self.n) * 1e8
        qf = -np.dot(Qf,desired)

        Qhf = np.zeros(shape=(self.n+1,self.n+1))
        Qhf[0:self.n,0:self.n] = Qf
        Qhf[0:self.n,[self.n]] = qf
        Qhf[[self.n],0:self.n] = qf.T
        Qhf[self.n,self.n] = np.dot(desired.T,np.dot(Qf,desired))

        (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                    B=B,
                                    Q=Q,
                                    R=R,
                                    ctdt='dt')
        assert np.allclose(pk,np.zeros(1))
        #pk should be zeros. 

        T = T+1
        Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)

        x_from_homo = np.zeros(self.n+1)
        x_from_homo[0:self.n] = x_from[0:self.n]
        x_from_homo[self.n] = 1
        #assert False
        return np.dot(x_from_homo.T,np.dot(Ps[0],x_from_homo))


    
    def distance_cache(self,from_node,to_point):
        #to_point is an array and from_point is a node

        global A
        global B
        global Q
        global R
        global max_time_horizon

        x_from = from_node['state']
        x_toward = to_point
        assert len(x_toward)==self.n+1

        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<0:
            return np.inf
        elif T==0:
            return 0 if self.same_state(x_from,x_toward) else np.inf
        
        if T < 5:
            #this technique isn't too accurate for short times due to slack in the final-value constraint
            #so do something else.
            return self.distance(from_node,to_point)
        
        if 'ctg' not in from_node:
            self.node_cache_ctg(from_node)

        if T >= len(from_node['ctg']):
            print 'requested uncached distance!!!'
            distance(from_node,to_point)

        ctg = from_node['ctg'][T]

        x_to_homo = np.zeros(self.n+1)
        x_to_homo[0:self.n] = x_toward[0:self.n]
        x_to_homo[self.n] = 1
         
        return np.dot(x_to_homo,np.dot(ctg,x_to_homo.T))

    def distance_direct_steer_cache(self,from_node,to_point):
        #print from_node['state'], to_point
        #to_point is an array and from_point is a node
        assert len(to_point)==self.n+1
        x_actual,action = self.steer_cache(from_node,to_point)
        if self.same_state(x_actual,to_point): #if actually drove there:
            return self.cost(from_node['state'],action)
        else:
            return np.inf

    def check_cache_distances(self,rrt,to_point):
        for node in rrt.tree.nodes():
            from_node = rrt.tree.node[node]
            d1 = self.distance(from_node,to_point)
            d2 = self.distance_cache(from_node,to_point)
            d3 = self.distance_direct(from_node,to_point)
            d4 = self.distance_direct_qp(from_node,to_point)
            d5 = self.distance_direct_steer_cache(from_node,to_point)
            T = to_point[self.n] - from_node['state'][self.n] 
            print d1,d2,d3,d4,d5,T
    #        if not (a == np.inf and b == np.inf):
    #            print a,b,abs(a-b)/(abs(a)+abs(b))




A = np.matrix([ [1,0],
                [1e-1,1] ])
B = np.matrix([ [1e-1],
                [0] ])

Q = np.zeros(shape=(2,2))
R = np.eye(1)

max_time_horizon = 50
goal = np.array([0,.8,50])

no_obstacles_test = False
def obstacles(x,y):
    if no_obstacles_test: return True

    return abs(x)<.48 #velocity limit
    #return true for if point is not in collision
    out_ball = (x-0)**2 + (y-0)**2 > .5**2
    in_square = np.logical_and(y>.1,np.logical_and(.3<x,x<.5))
    in_square1 = np.logical_and(np.logical_and(.6<x,x<.99),np.logical_and(.4<y,y<.8))
    in_field_x = np.logical_and(np.logical_and(-1<=x,x<=1),np.logical_and(-1<=y,y<=1))
    return np.logical_and(in_field_x, out_ball)
    return np.logical_and(out_ball, 
                          np.logical_and(np.logical_not(in_square),
                                         np.logical_not(in_square1)))
    
def isStateValid(state):
    assert len(state) == 3
    x = state[0]
    y = state[1]
    t = state[2]
    return t<= max_time_horizon and bool(obstacles(x,y))    #time obstacle here. prevents nodes from getting pushed farther and farther into time by rewiring

def isActionValid(action):
    return True
    r = abs(action[0]) < 1e6
    if not r: print 'action constraint!',action
    return r

def action_state_valid(x,u):
    return isStateValid(x) and isActionValid(u)

def goal_test(node):
    goal_region_radius = .01
    n = 2
    return np.sum(np.abs(node['state'][0:n]-goal[0:n])) < goal_region_radius #disregards time
    return distance(node,goal) < goal_region_radius                     #need to think more carefully about this one
    
def distance_from_goal(node):
    return 0


lqr_rrt = LQR_RRT(A,B,Q,R,max_time_horizon)
lqr_rrt.action_state_valid = action_state_valid


def sample():
    if np.random.rand()<.9:
        statespace = np.random.rand(2)*np.array([.4,2])-np.array([.2,1])
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
    

start = np.array([0,-1,0])
rrt = RRT(state_ndim=3,control_ndim=1)

rrt.goal = goal

rrt.set_distance(lqr_rrt.distance_cache)
rrt.set_same_state(lqr_rrt.same_state)
rrt.set_cost(lqr_rrt.cost)
rrt.set_steer(lqr_rrt.steer_cache)

rrt.set_goal_test(goal_test)
rrt.set_sample(sample)
#rrt.set_collision_check(isStateValid)
rrt.set_collision_free(lqr_rrt.collision_free)

rrt.set_distance_from_goal(distance_from_goal)

rrt.gamma_rrt = 4.0
rrt.eta = 0.2
rrt.c = 1

rrt.set_start(start)
rrt.init_search()


def plot_tree(rrt):
    from mayavi import mlab
    tree = rrt.tree
    leafs = [i for i in tree.nodes() if len(tree.successors(i)) == 0]
    accounted = set()
    paths = []

    for leaf in leafs:
        this_node = leaf
        this_path = []
        paths.append(this_path)
        while True:
            this_path.append(this_node)
            accounted.add(this_node)
            p = tree.predecessors(this_node)
            if len(p) == 0:
                break
            assert len(p) == 1
            this_node = p[0]
            if this_node in accounted:
                this_path.append(this_node) #repeat branching node
                break

    for p in paths:
        s = np.array([rrt.tree.node[i]['state'] for i in p])
        c = np.array([rrt.tree.node[i]['cost'] for i in p])
        mlab.plot3d(s[:,0],s[:,1],s[:,2]/50,c,tube_radius=0.002,colormap='Spectral')

if __name__ == '__main__':
    if False:
        import shelve
        import os.path
        p = 'di_rrt_66k.shelve'
        assert os.path.exists(p)
        load_shelve = shelve.open(p)
        rrt.load(load_shelve)
        
    import copy

    interactive_rrt = copy.deepcopy(rrt)

    x = np.linspace(-1,1,1000)
    X,Y = np.meshgrid(x,x)
    obstacle_bitmap = obstacles(X,Y) #rasterize the obstacles

    #ugly globalness
    info_text = None

    def draw_rrt(rrt,int_ax,control_ax=None):
        global no_obstacles_test
        global obstacle_bitmap
        global info_text

        ani_ax = int_ax
        ani_rrt = rrt
        
        ani_ax.cla()
        ani_ax.set_xlim(-1,1)
        ani_ax.set_ylim(-1.5,1)
        #ani_ax.set_aspect('equal')
        #ani_ax.set_aspect('auto')
        if not no_obstacles_test: ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5,zorder=1,aspect='auto')    
        
        for l in ani_ax.lines:
            l.remove()
        for p in ani_ax.patches:
            p.remove()
            
        tree = ani_rrt.tree
        
        if (ani_rrt.viz_change):
            #viz_x_nearest = tree.node[ani_rrt.viz_x_nearest_id]['state']
            #viz_x_new = tree.node[ani_rrt.viz_x_new_id]['state']
            #viz_x_from = tree.node[ani_rrt.viz_x_from_id]['state']
            
            viz_x_nearest = ani_rrt.viz_x_nearest
            viz_x_new = ani_rrt.viz_x_new
            viz_x_from =ani_rrt.viz_x_from
        
        best_sol = ani_rrt.best_solution(goal) 
        #xpath = np.array([tree.node[i]['state'] for i in best_sol]).T #straight line connection between nodes
        xpath = [tree.node[best_sol[0]]['state']]
        for (node_from,node_to) in zip(best_sol[:-1],best_sol[1:]):
            xpath.extend( lqr_rrt.run_forward(tree.node[node_from]['state'],tree.node[node_to]['action']) )
        
        xpath = np.array(xpath).T
        ani_ax.plot(xpath[0],xpath[1],ls='--',lw=10,alpha=.7,color=(.2,.2,.2,1),zorder=2,label='best path so far')

        if control_ax is not None:
            upath = rrt.get_action(best_sol)
            control_ax.cla()
            control_ax.plot(np.squeeze(upath))
        
        #draw paths that collided
        if not rrt.viz_collided_paths is None:
            lines = []
            for (node,action) in rrt.viz_collided_paths:
                x0 = node['state']
                xs = lqr_rrt,run_forward(x0,action)
                xs = np.concatenate((x0.reshape((1,-1)),xs))
                lines.append(xs[:,0:2])
            
            collision_collection = mpl.collections.LineCollection(lines,linewidths=1,linestyles='solid')
            collision_collection.set_color('red')
            ani_ax.add_collection(collision_collection)
            collision_collection.set_zorder(4)

            rrt.viz_collided_paths = []                


        if (ani_rrt.viz_change):
            #draws a straight edge
            ani_ax.plot([viz_x_from[0],viz_x_new[0]],[viz_x_from[1],viz_x_new[1]],'y',lw=5,alpha=.7,zorder=3,label='new extension')
            

        pos = {n:tree.node[n]['state'][0:2] for n in tree.nodes()}
        col = [tree.node[n]['cost'] for n in tree.nodes()]

        int_ax.get_figure().sca(int_ax) #set the current axis to the int_ax. there is some bug in networkx/matplotlib
        node_collection = nx.draw_networkx_nodes(G=tree,
                                                pos=pos,
                                                ax=ani_ax,
                                                node_size=25,
                                                node_color=col,
                                                cmap = mpl.cm.get_cmap(name='copper'),
                                                )

        if not node_collection is None:
            node_collection.set_zorder(5)

        if False:                                        
            #draw straight edges
            edge_collection = nx.draw_networkx_edges(G=tree,
                                                    pos=pos,
                                                    ax=ani_ax,
                                                    edge_color='b',
                                                    )
        else:
            #draw dynamical edges
            lines = []
            for i in tree.nodes():
                s = tree.predecessors(i)
                if len(s) == 0:
                    continue
                assert len(s) == 1 #it's a tree
                s = s[0]
                x0 = tree.node[s]['state']
                xs = run_forward(x0, tree.node[i]['action'])
                xs = np.concatenate((x0.reshape((1,-1)),xs))
                lines.append(xs[:,0:2])
            edge_collection = mpl.collections.LineCollection(lines)
            ani_ax.add_collection(edge_collection)
        
        if not edge_collection is None:
                edge_collection.set_zorder(4)            
        
        #mfc, mec, mew is marker face color, edge color, edge width
        if (ani_rrt.viz_change):

            #ani_ax.add_patch(mpl.patches.Circle(xy=viz_x_new,radius=ani_rrt.viz_search_radius,
            #                                    alpha=.3,fc='none',ec='b',label='_rewire radius'))
        
            ani_ax.plot(*ani_rrt.viz_x_rand[0:2],marker='*', mfc='k', mec='k', ls='None', zorder=6, label='x_rand')
            ani_ax.plot(*viz_x_nearest[0:2],marker='p', mfc='c', mec='c', ls='None', zorder=7, ms=5, label='x_nearest')
            ani_ax.plot(*viz_x_new[0:2], marker='x', mfc='r', mec='r', ls='None', zorder=8, label='x_new')    
            ani_ax.plot(*viz_x_from[0:2], marker='o', mfc='g',mec='g', ls='None',alpha=.5, zorder=9, label='x_from')
            
            if ani_rrt.viz_x_near is not None and len(ani_rrt.viz_x_near)>0:
                x_near = np.array(ani_rrt.viz_x_near)
                ani_ax.plot(x_near[:,0],x_near[:,1], marker='o', mfc='none',mec='r', mew=1 ,ls='None',alpha=.5, zorder=10, label='X_near')

        
        ani_ax.legend(bbox_to_anchor=(1.05,0.0),loc=3,
                       ncol=1, borderaxespad=0.,
                        fancybox=True, shadow=True,numpoints=1)
        
        #ani_ax.legend()
        plt.setp(ani_ax.get_legend().get_texts(),fontsize='small')

        info = ""
        info += "# nodes: %d\n" % (len(tree.nodes()))
        info += "# edges: %d\n" % (len(tree.edges()))
        info += "cost: %s\n" % (str(ani_rrt.worst_cost) if ani_rrt.found_feasible_solution else "none")

        if info_text is not None:
            info_text.set_text(info)
        else:   
            info_text = ani_ax.figure.text(.8, .5, info,size='small')

    if True:
        int_fig = plt.figure(None)
        int_ax = int_fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=.15,bottom=.35,right=.8)
        action_ts_ax = plt.axes([.15,.2,.65,.1])
        time_slider_ax = plt.axes([.15,.1,.65,.03])
        
        time_slider = mpl.widgets.Slider(time_slider_ax,'Time',0,max_time_horizon,valinit=1)
        
        # shift the axis to make room for legend
        #box = int_ax.get_position()
        #int_ax.set_position([box.x0-.1, box.y0+.15, box.width, box.height])
        #plt.tight_layout()
        draw_rrt(interactive_rrt,int_ax,action_ts_ax)        
        
        #sampler = lambda : np.array([1.0,1.0])
        #interactive_rrt.set_sample(sampler)
        #interactive_rrt.search(1)
        
        def rrts(xrand):
            xrand = np.array(xrand)
            interactive_rrt.set_sample(lambda : xrand)
            interactive_rrt.search(1)
            draw_rrt(interactive_rrt,int_ax,action_ts_ax)
            interactive_rrt.viz_change = False
            int_fig.canvas.draw()
                
        def button_press_event_dispatcher(event):
            if int_fig.canvas.widgetlock.locked(): #matplotlib widget in use
                return

            if event.inaxes is not int_ax:
                return

            if event.button == 1: #sample 
                interactive_T = int(time_slider.val)
                print 'interactive_T',interactive_T

                #interactive_T = sample()[2]

                p = np.array([event.xdata,event.ydata,interactive_T])
                print 'sample',p

                interactive_rrt.extend(p)
                draw_rrt(interactive_rrt,int_ax,action_ts_ax)
                interactive_rrt.viz_change = False
                int_fig.canvas.draw()

                upath = []
                for i in interactive_rrt.best_solution(goal)[1:]:
                    upath.extend(interactive_rrt.tree.node[i]['action'])
                upath = np.array(upath)
                print np.squeeze(upath)             

            elif event.button == 3: #print node info on right click.
                #node_id, distance = interactive_rrt.nearest_neighbor([event.xdata,event.ydata,interactive_T])
                pos = np.array([interactive_rrt.tree.node[i]['state'][0:2] for i in interactive_rrt.tree.nodes()])
                distances = np.sum( (pos - np.array([event.xdata,event.ydata]))**2,axis=1)
                closest = np.argmin(distances)
                node_id = closest            
                
                state = interactive_rrt.tree.node[node_id]['state']
                int_ax.text(state[0],state[1],s=str(node_id),zorder=30)    #text on top
                int_fig.canvas.draw()
                
                print node_id, interactive_rrt.tree.node[node_id]
                
            import sys
            sys.stdout.flush() #function is called asyncrhonously, so any print statements might not flush
                
        int_fig.canvas.mpl_connect('button_press_event', button_press_event_dispatcher)    
        
       
    ani_rrt = copy.deepcopy(rrt)

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


    if False:
        ani_fig = plt.figure(None)
        ani_ax = ani_fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=.15,bottom=.35,right=.8)
        action_ts_ax = plt.axes([.15,.2,.65,.1])
        time_slider_ax = plt.axes([.15,.1,.65,.03])
        
        time_slider = mpl.widgets.Slider(time_slider_ax,'Time',0,max_time_horizon,valinit=1)


    #    ani_ax = ani_fig.gca()
        
    #    ani_ax.set_xlim(-1,1)
    #    ani_ax.set_ylim(-1,1)
    #    ani_ax.set_aspect('equal')
        
        #ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5,aspect='auto')    
        
        #import copy
        
        # shift the axis to make room for legend
    #    box = ani_ax.get_position()
    #    ani_ax.set_position([box.x0-.1, box.y0, box.width, box.height])
        
        worst_costs = []

        saved_frame = []

        def update_frame(i): 
            print 'frame: ',i
            ani_rrt.force_iteration(quiet=False)
            ani_ax.set_title('time index: %d'%(i))
            draw_rrt(ani_rrt,ani_ax,action_ts_ax)
            time_slider.set_val(ani_rrt.viz_x_rand[2])
            global worst_costs
            worst_costs.append(ani_rrt.worst_cost)

            if(i%50==0):
                global saved_frame
                if i not in saved_frame:
                    import shelve
                    s = shelve.open('di_rrt_%04d.shelve'%i)
                    ani_rrt.save(s)
                    s.close()
            
           
        ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=10000,interval=500)
        ani.save('di_rrt.mp4', fps=5, codec='mpeg4', clear_temp=False)
        #ani.save('test.mp4', fps=20, codec='mpeg4', clear_temp=True)
        
        


