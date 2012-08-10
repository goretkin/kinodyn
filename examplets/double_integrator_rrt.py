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

from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR

A = np.matrix([ [1,0],
                [1e-1,1] ])
B = np.matrix([ [1e-1],
                [0] ])

Q = np.zeros(shape=(2,2))
R = np.eye(1)

def run_forward(x0,us):
    global A
    global B
    n = 2+1  #+1 for time dimension 
    m = 1
    us = np.reshape(us,newshape=(-1,m))
    assert len(x0) == n
    T = us.shape[0]
    xs = np.zeros(shape=(T+1,n))
    xs[0] = np.squeeze(x0)

    for i in range(1,T+1):
        xs[i,0:2] = np.dot(A,xs[i-1,0:2].T) + np.dot(B,us[i-1].T)
        xs[i,2] = xs[i-1,2] + 1
    return xs[1:]

def run_forward_fb(x0,gain_schedule):
    global A
    global B
    n = 2+1  #+1 for time dimension 
    m = 1
    assert len(x0) == n
    T = gain_schedule.shape[0]
    assert gain_schedule.shape[2] == n-1 #no time in the gain 
    assert gain_schedule.shape[1] == m

    xs = np.zeros(shape=(T+1,n))
    us = np.zeros(shape=(T,m))

    xs[0] = np.squeeze(x0)

    for i in range(1,T+1):
        us[i-1] = -1* np.dot(gain_schedule[i-1],xs[i-1,0:2])
        xs[i,0:2] = np.dot(A,xs[i-1,0:2].T) + np.dot(B,us[i-1].T)
        xs[i,2] = xs[i-1,2] + 1
    return xs[1:],us

no_obstacles_test = False
def obstacles(x,y):
    if no_obstacles_test: return True
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
    x = state[0]
    y = state[1]
    assert len(state) == 3
    return bool(obstacles(x,y))

def isActionValid(action):
    return abs(action[0]) < .5

max_time_horizon = 100
goal = np.array([1.0,1.0,100])
def sample():
    if np.random.rand()<.9:
        statespace = np.random.rand(2)*2-1
        #time = np.random.randint(0,max_time_horizon,size=1) + 1
        time = np.array(min(np.random.geometric(.06,size=1),max_time_horizon))
        time = np.reshape(time,newshape=(1,))
        return np.concatenate((statespace,time))
    else: #goal bias
        return goal

def collision_free(from_node,action):
    """
    check that taking action from from_node produces a collision free trajectory
    if not, return a partial trajectory for the state (x_path) and control (u_path)
    u_path is a list of actions -- it partitions the actions.
    """
    x0 = from_node['state']
    x_path = []
    u_path = []
    all_the_way = True
    if isStateValid(x0):
        x_path_np = run_forward(x0,action)
        for (x,u) in zip(x_path_np,action):
            if not isStateValid(x) or not isActionValid(u):
                all_the_way = False
                break
            x_path.append(x)
            u_path.append(np.array([u])) #brackets keep the elements of u_path looking like an action

    return x_path, u_path, all_the_way        
 
def cost(x_from,action):
    #this does not include the cost of being in x_from
    global Q
    global R
    assert len(x_from) == 3
    assert action.shape[1] == 1
    x_path = run_forward(x_from,action)
    cost = 0
    for i in range(action.shape[0]):
        x = x_path[[i],0:2].T #don't include time
        u = action[[i],:].T
        cost += np.squeeze( np.dot(x.T,np.dot(Q,x))  + 
                            np.dot(u.T,np.dot(R,u))
                            ) 
    return cost

def steer(x_from_node,x_toward):
    global A
    global B
    global Q
    global R

    x_from = x_from_node['state']
    assert len(x_from) == 3
    T = x_toward[2] - x_from[2] #how much time to do the steering

    assert T-int(T) == 0 #discrete time

    T=int(T)
    
    if T<=0:
        return (x_from,np.zeros(shape=(0,1)))   #stay here


    desired = np.matrix([[x_toward[0]],
                         [x_toward[1]]])                     

    Qf = np.eye(2) * 1e8
    qf = np.dot(Qf,desired)

    Qhf = np.zeros(shape=(3,3))
    Qhf[0:2,0:2] = Qf
    Qhf[0:2,[2]] = qf
    Qhf[[2],0:2] = qf.T
    Qhf[2,2] = np.dot(desired.T,np.dot(Qf,desired))

    (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                B=B,
                                c=np.zeros(shape=(2,1)),
                                Q=Q,
                                R=R,
                                q=np.zeros(shape=(2,1)),
                                r=np.zeros(1),
                                ctdt='dt')
    #pk should be zeros. 

    Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)

    xs = np.zeros(shape=(T+1,3))
    us = np.zeros(shape=(T,1))
    xs[0] = x_from

    for i in range(T):
        us[i] = -1 * np.dot(Fs[i,:,0:2],xs[i,0:2]) + Fs[i,:,2]
        xs[i+1,0:2] = np.dot(A,xs[i,0:2].T) + np.dot(B,us[i].T)
        xs[i+1,2] = xs[i,2] + 1
    x_actual = xs[-1]    
    
    return (x_actual, us)

def steer_QP(x_from_node,x_toward):
    global A
    global B
    global Q
    global R

    x_from = x_from_node['state']
    assert len(x_from) == 3
    T = x_toward[2] - x_from[2] #how much time to do the steering

    assert T-int(T) == 0 #discrete time

    T=int(T)
    
    if T<=0:
        return (x_from,np.zeros(shape=(0,1)))   #stay here

    try:
        qpsol, qpmats ,xs,us = LQR_QP(A,B,Q,R,(T+1),
                            x0=x_from[0:2],
                            xT=x_toward[0:2])
    except ValueError as e:
        #quadratic program is probably infeasible. This can happen if the time horizon is too short and the system's reachability doesn't include the final-value constraint
        return (x_from,np.zeros(shape=(0,1)))   #stay here (could do something smarter)
        
    (QP_P,QP_q,QP_A,QP_B) = qpmats

    xs = xs.T
    us = us.T
    x_actual = np.concatenate((     xs[-1],
                                    [x_from[2]+us.shape[0]]
                            ))
    return (x_actual, us)


    
def distance_direct(from_node,to_point):
    #print from_node['state'], to_point
    #to_point is an array and from_point is a node
    assert len(to_point)==3
    x_actual,action = steer(from_node,to_point)
    if np.allclose(x_actual,to_point,atol=1e-4): #if actually drove there:
        return cost(from_node['state'],action)
    else:
        return np.inf

def distance_direct_qp(from_node,to_point):
    #print from_node['state'], to_point
    #to_point is an array and from_point is a node
    assert len(to_point)==3
    x_actual,action = steer_QP(from_node,to_point)
    if np.allclose(x_actual,to_point,atol=1e-4): #if actually drove there:
        return cost(from_node['state'],action)
    else:
        return np.inf

def distance(from_node,to_point):
    #to_point is an array and from_point is a node

    global A
    global B
    global Q
    global R

    x_from = from_node['state']
    x_toward = to_point
    assert len(x_toward)==3

    T = x_toward[2] - x_from[2] #how much time to do the steering

    assert T-int(T) == 0 #discrete time

    T=int(T)
    
    if T<0:
        return np.inf
    elif T==0:
        return 0 if np.allclose(x_from,x_toward) else np.inf

    desired = np.matrix([[x_toward[0]],
                         [x_toward[1]]])                     

    #we want the final bowl to be centered at desired:
    #(x-x_d)^T * Qf * (x-x_d)
    #xT*Qf*x -x_dT * Qf * x - xT *Qf *x_d * x_dT * Qf * x_d
    Qf = np.eye(2) * 1e6
    qf = np.dot(Qf,desired)

    Qhf = np.zeros(shape=(3,3))
    Qhf[0:2,0:2] = Qf
    Qhf[0:2,[2]] = qf
    Qhf[[2],0:2] = qf.T
    Qhf[2,2] = np.dot(desired.T,np.dot(Qf,desired))

    (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                B=B,
                                Q=Q,
                                R=R,
                                ctdt='dt')
    assert np.allclose(pk,np.zeros(1))
    #pk should be zeros. 

    Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)

    x_from_homo = np.zeros(3)
    x_from_homo[0:2] = x_from[0:2] - x_toward[0:2]
    x_from_homo[2] = 1
    #assert False
    return np.dot(x_from_homo,np.dot(Ps[0],x_from_homo.T))


goal_region_radius = .01
def goal_test(node):
    global goal
    return np.sum(np.abs(node['state'][0:2]-goal[0:2])) < goal_region_radius #disregards time
    return distance(node,goal) < goal_region_radius                     #need to think more carefully about this one
    
def distance_from_goal(node):
    global goal
    return 0#max(distance(node,goal)-goal_region_radius,0)

start = np.array([0,-1,0])
rrt = RRT(state_ndim=3)

rrt.set_distance(distance)
rrt.set_cost(cost)
rrt.set_steer(steer)

rrt.set_goal_test(goal_test)
rrt.set_sample(sample)
rrt.set_collision_check(isStateValid)
rrt.set_collision_free(collision_free)

rrt.set_distance_from_goal(distance_from_goal)

rrt.gamma_rrt = 4.0
rrt.eta = 0.5
rrt.c = 1

rrt.set_start(start)
rrt.init_search()

if __name__ == '__main__':
    if False:
        import shelve
        #load_shelve = shelve.open('examplets/rrt_2d_example.shelve')
        load_shelve = shelve.open('rrt_0950.shelve')
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
        ani_ax.set_ylim(-1,1)
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
        xpath = np.array([tree.node[i]['state'] for i in best_sol]).T
        ani_ax.plot(xpath[0],xpath[1],ls='--',lw=10,alpha=.7,color=(.2,.2,.2,1),zorder=2,label='best path so far')

        if control_ax is not None:
            control_ax.cla()
            upath = np.array([tree.node[i]['action'] for i in best_sol[1:]]).T  #the first action is None -- action at the root
            print upath.shape
            control_ax.plot(np.squeeze(upath))

        if (ani_rrt.viz_change):
            ani_ax.plot([viz_x_from[0],viz_x_new[0]],[viz_x_from[1],viz_x_new[1]],'y',lw=5,alpha=.7,zorder=3,label='new extension')

        pos=[tree.node[n]['state'][0:2] for n in tree.nodes()]

        int_ax.get_figure().sca(int_ax) #set the current axis to the int_ax. there is some bug in networkx/matplotlib
        node_collection = nx.draw_networkx_nodes(G=tree,
                                                pos=pos,
                                                ax=ani_ax,
                                                node_size=25,
                                                node_color=[tree.node[n]['cost'] for n in tree.nodes()],
                                                cmap = mpl.cm.get_cmap(name='copper'),
                                                )
        
        edge_collection = nx.draw_networkx_edges(G=tree,
                                                pos=pos,
                                                ax=ani_ax,
                                                edge_color='b',
                                                )
        if not edge_collection is None:
            edge_collection.set_zorder(4)
            
        if not node_collection is None:
            node_collection.set_zorder(5)
        
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
                sampler = lambda : p
                
                interactive_rrt.set_sample(sampler)
                interactive_rrt.search(1)
                draw_rrt(interactive_rrt,int_ax,action_ts_ax)
                interactive_rrt.viz_change = False
                int_fig.canvas.draw()

                upath = []
                for i in interactive_rrt.best_solution(goal)[1:]:
                    upath.append(interactive_rrt.tree.node[i]['action'])
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
        
        ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5,aspect='auto')    
        
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
        
        


