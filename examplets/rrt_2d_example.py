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

def collision_free(from_node,action):
    """
    check that taking action from from_node produces a collision free trajectory
    if not, return a partial trajectory for the state (x_path) and control (u_path)
    """
    assert len(action.shape) == 2 #two dimensions (array of actions)
    assert action.shape[1] == 2 #action is two dimensional

    x_path = [from_node['state']]       #initialize this with the from_node, but when return, make sure to take it out.
    u_path = []
    all_the_way = False

    if isStateValid(from_node['state']):
        last_x_final = from_node['state'] 
        for single_action in action:
            #x_path.append(from_node['state'])
            x_final = last_x_final + single_action

            step = .1

            for i in itertools.count():
                u = x_final - x_path[i] #actuation to go to x_final
                if np.linalg.norm(u) < 1e-6:
                    all_the_way = True
                    break                
                if np.linalg.norm(u) > step:
                    u = u / np.linalg.norm(u) * step
                x_next = x_path[i] + u

                if not isStateValid(x_next):
                    break
                u_path.append(u)
                x_path.append(x_next)
            last_x_final = x_final
    x_path = np.array(x_path[1:])
    u_path = np.array(u_path)

    return x_path, u_path, all_the_way    

def cost(x_from,action):
    #cost is the Euclidian length of the path.
    assert len(x_from) == 2
    assert len(action.shape) == 2
    assert action.shape[1] == 2
    return sum ( [np.linalg.norm(a) for a in action] ) 

def steer(x_from_node,x_toward):
    x_from = x_from_node['state']
    extension_direction = x_toward-x_from
    norm = np.linalg.norm(extension_direction)
    if norm > .5:
        extension_direction = extension_direction/norm
        extension_direction *= .5
    control = extension_direction #steer
    
    x_new = x_from + control 
    return (x_new,control.reshape((1,-1)))
    
def distance(from_node,to_point):
    assert len(to_point)==2
    #to_point is an array and from_point is a node
    return np.linalg.norm(to_point-from_node['state'])

goal_region_radius = .05
def goal_test(node):
    global goal
    return distance(node,goal) < goal_region_radius
    
def distance_from_goal(node):
    global goal
    return max(distance(node,goal)-goal_region_radius,0)

start = np.array([-1,-1])*1    
rrt = RRT(state_ndim=2)

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


rrt.goal = goal
rrt.set_start(start)
rrt.init_search()

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

def draw_voronoi(ax,rrt):
    xr = ax.get_xlim()
    yr = ax.get_xlim()
    
    xres = 500
    yres = 500

    xs= np.linspace(xr[0],xr[1],xres)
    ys= np.linspace(yr[0],yr[1],yres)
    
    grid = np.array(np.meshgrid(xs,ys))
    grid = grid.reshape((2,-1))
    grid = grid.T
    #grid is a 2-by-(xres * yres) array. we want the nearest node for each of those (xres*yres) points

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(algorithm='kd_tree',n_neighbors=1)
    nodes = np.array(rrt.tree.nodes())
    states = np.array([rrt.tree.node[i]['state'] for i in nodes])
    nn.fit(states,nodes)
    nn_res = nn.kneighbors(grid,return_distance=False)
    nn_res = nn_res.reshape((xres,yres))
    nn_res = np.array(nn_res,dtype=np.float)
    
    #ns_res is an xres-by-yres array and contains the node-id of the nearest neighbor at each [i,j]
    print 'regions' , np.unique(nn_res)
    if np.max(nn_res)> 0: nn_res /= float(np.max(nn_res))   #normalize for color map. this is a stupid way to assign colors to regions
    ax.imshow(nn_res,origin='lower',extent=[xr[0],xr[1],yr[0],yr[1]],alpha=.5,zorder=2,cmap=mpl.cm.get_cmap(name='prism'))    
    ax.figure.canvas.draw()


def draw_rrt(int_ax,rrt):
    global obstacle_bitmap
    global info_text

    ani_ax = int_ax
    ani_rrt = rrt
    
    ani_ax.cla()
    ani_ax.set_xlim(-1,1)
    ani_ax.set_ylim(-1,1)
    ani_ax.set_aspect('equal')    
    ani_ax.imshow(obstacle_bitmap,origin='lower',extent=[-1,1,-1,1],alpha=.5,zorder=1)    
    
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
        
    xpath = np.array([tree.node[i]['state'] for i in ani_rrt.best_solution(goal)]).T
    ani_ax.plot(xpath[0],xpath[1],ls='--',lw=10,alpha=.7,color=(.2,.2,.2,1),zorder=2,label='best path so far')

    if (ani_rrt.viz_change):
        ani_ax.plot([viz_x_from[0],viz_x_new[0]],[viz_x_from[1],viz_x_new[1]],'y',lw=5,alpha=.7,zorder=3,label='new extension')

    pos=nx.get_node_attributes(tree,'state')
    
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
    
    #mfc is marker face color
    if (ani_rrt.viz_change):
        ani_ax.add_patch(mpl.patches.Circle(xy=viz_x_new,radius=ani_rrt.viz_search_radius,
                                            alpha=.3,fc='none',ec='b',label='_rewire radius'))
    
        ani_ax.plot(*ani_rrt.viz_x_rand,marker='*', mfc='k', mec='k', ls='None', zorder=6, label='x_rand')
        ani_ax.plot(*viz_x_nearest,marker='p', mfc='c', mec='c', ls='None', zorder=7, ms=5, label='x_nearest')
        ani_ax.plot(*viz_x_new, marker='x', mfc='r', mec='r', ls='None', zorder=8, label='x_new')    
        ani_ax.plot(*viz_x_from, marker='o', mfc='g',mec='g', ls='None',alpha=.5, zorder=9, label='x_from')

        if ani_rrt.viz_x_near is not None and len(ani_rrt.viz_x_near)>0:
            x_near = np.array(ani_rrt.viz_x_near)
            ani_ax.plot(x_near[:,0],x_near[:,1], marker='o', mfc='none',mec='r', mew=1 ,ls='None',alpha=.5, zorder=10, label='X_near')


    ani_ax.legend(bbox_to_anchor=(1.05,0.0),loc=3,
                   ncol=1, borderaxespad=0.,
                    fancybox=True, shadow=True,numpoints=1)

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
    frame_counter = 0

    int_fig = plt.figure(None)
    int_ax = int_fig.add_subplot(111)
    
    # shift the axis to make room for legend
    box = int_ax.get_position()
    int_ax.set_position([box.x0-.1, box.y0, box.width, box.height])
    
    draw_rrt(int_ax,interactive_rrt)
    
    
    #sampler = lambda : np.array([1.0,1.0])
    #interactive_rrt.set_sample(sampler)
    #interactive_rrt.search(1)
    
    #draw_rrt(int_ax,interactive_rrt)
    
    def save_frame(fig):
        global frame_counter
        s ='int_rrt_2d_%03d.png'%(frame_counter)
        fig.savefig(s)
        frame_counter += 1

    def rrts(xrand):
        xrand = np.array(xrand)
        interactive_rrt.set_sample(lambda : xrand)
        interactive_rrt.search(1)
        draw_rrt(int_ax,interactive_rrt)
        interactive_rrt.viz_change = False
        int_fig.canvas.draw()
            
    def button_press_event_dispatcher(event):
        if int_fig.canvas.widgetlock.locked(): #matplotlib widget in use
            return
        if event.xdata is None or event.ydata is None: #make sure clicked on axes
            return
            
        if event.button == 1: #sample 
            p = np.array([event.xdata,event.ydata])
            
            sampler = lambda : p
            
            interactive_rrt.set_sample(sampler)
            interactive_rrt.search(1)
            draw_rrt(int_ax,interactive_rrt)
            interactive_rrt.viz_change = False
            int_fig.canvas.draw()
            save_frame(int_fig)

        elif event.button == 3: #print node info / draw voronoi regions
            draw_voronoi(int_ax,interactive_rrt)
            node_id, distance = interactive_rrt.nearest_neighbor([event.xdata,event.ydata])
            state = interactive_rrt.tree.node[node_id]['state']
            int_ax.text(*state,s=str(node_id),zorder=30)    #text on top
            int_fig.canvas.draw()
            
            print node_id, interactive_rrt.tree.node[node_id]
            
        import sys
        sys.stdout.flush() #function is called asyncrhonously, so any print statements might not flush
            
    int_fig.canvas.mpl_connect('button_press_event', button_press_event_dispatcher)    
    
    import shelve
    plt.show()
    #for s in shelve.open('kin2d_rewire_bug.shelve')['sample_history']:
    #    rrts(s)
   
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


if False:
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
    
    worst_costs = []

    saved_frame = []

    def update_frame(i): 
        print 'frame: ',i
        ani_rrt.force_iteration()
        ani_ax.set_title('time index: %d'%(i))
        draw_rrt(ani_ax,ani_rrt)

        global worst_costs
        worst_costs.append(ani_rrt.worst_cost)

        if(i%50==0):
            global saved_frame
            if i not in saved_frame:
                import shelve
                s = shelve.open('rrt_%04d.shelve'%i)
                ani_rrt.save(s)
                s.close()
        
       
    ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=1500,interval=500)
    ani.save('rrt.mp4', fps=5, codec='mpeg4', clear_temp=False)
    #ani.save('test.mp4', fps=20, codec='mpeg4', clear_temp=True)
    
    


