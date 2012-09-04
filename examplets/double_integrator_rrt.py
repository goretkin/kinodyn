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

from lqr_rrt import LQR_RRT


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

rrt.sample_goal = lambda : goal

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

def draw_rrt(rrt,int_ax,control_ax=None,plot_dims=[0,1]):
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
    ani_ax.plot(xpath[plot_dims[0]],xpath[plot_dims[1]],ls='--',lw=10,alpha=.7,color=(.2,.2,.2,1),zorder=20,label='best path so far')

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
            lines.append(xs[:,plot_dims])
        
        collision_collection = mpl.collections.LineCollection(lines,linewidths=1,linestyles='solid')
        collision_collection.set_color('red')
        ani_ax.add_collection(collision_collection)
        collision_collection.set_zorder(4)

        rrt.viz_collided_paths = []                


    if (ani_rrt.viz_change):
        #draws a straight edge
        new_ext_x = [viz_x_from[plot_dims[0]],viz_x_new[plot_dims[0]]]
        new_ext_y = [viz_x_from[plot_dims[1]],viz_x_new[plot_dims[1]]]
        ani_ax.plot(new_ext_x,new_ext_y,'y',lw=5,alpha=.7,zorder=3,label='new extension')
        

    pos = {n:tree.node[n]['state'][plot_dims] for n in tree.nodes()}
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
            lines.append(xs[:,plot_dims])
        edge_collection = mpl.collections.LineCollection(lines)
        ani_ax.add_collection(edge_collection)
    
    if not edge_collection is None:
            edge_collection.set_zorder(4)            
    
    #mfc, mec, mew is marker face color, edge color, edge width
    if (ani_rrt.viz_change):

        #ani_ax.add_patch(mpl.patches.Circle(xy=viz_x_new,radius=ani_rrt.viz_search_radius,
        #                                    alpha=.3,fc='none',ec='b',label='_rewire radius'))
    
        ani_ax.plot(*ani_rrt.viz_x_rand[plot_dims],marker='*', mfc='k', mec='k', ls='None', zorder=6, label='x_rand')
        ani_ax.plot(*viz_x_nearest[plot_dims],marker='p', mfc='c', mec='c', ls='None', zorder=7, ms=5, label='x_nearest')
        ani_ax.plot(*viz_x_new[plot_dims], marker='x', mfc='r', mec='r', ls='None', zorder=8, label='x_new')    
        ani_ax.plot(*viz_x_from[plot_dims], marker='o', mfc='g',mec='g', ls='None',alpha=.5, zorder=9, label='x_from')
        
        if ani_rrt.viz_x_near is not None and len(ani_rrt.viz_x_near)>0:
            x_near = np.array(ani_rrt.viz_x_near)
            ani_ax.plot(x_near[:,plot_dims[0]],x_near[:,plot_dims[1]], marker='o', mfc='none',mec='r', mew=1 ,ls='None',alpha=.5, zorder=10, label='X_near')

    
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
        
        


