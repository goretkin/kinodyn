import examplets_import

from rrt import RRT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import networkx as nx
import shelve


from double_integrator_rrt import rrt

try:
    rrt.load(shelve.open('di_rrt_0100.shelve'))
except AssertionError as e:
    print e

nodes = [n for n in rrt.tree.nodes()]

states = np.array([rrt.tree.node[i]['state'][:] for i in nodes])
costs = np.array([rrt.tree.node[i]['cost'] for i in nodes])

inds = np.argsort(states[:,2])

max_time = states[inds[-1],2]
max_cost = np.max(costs)

costs /= max_cost #normalize all costs



ani_fig = plt.figure(None)
ani_ax = ani_fig.add_subplot(1,1,1)
ani_ax.set_xlim(-1,1)
ani_ax.set_ylim(-1,1)

plot_times = np.arange(0,max_time,1)

cmap = mpl.cm.get_cmap(name='copper')

nodes_already_plotted = set()

def update_frame(i): 
    # no fade out
    ani_ax.clear()
    ani_ax.set_xlim(-1,1)
    ani_ax.set_ylim(-1,1)

    global nodes_already_plotted
    print 'frame: ',i
    t = plot_times[i]
    inds = np.where(states[:,2] <= t)[0]

    nodes_to_plot = set([nodes[i] for i in inds])
    nodes_to_plot = nodes_to_plot.difference(nodes_already_plotted)

    ani_ax.set_title('time index: %d'%(t))
    nodes_to_plot_list = list(nodes_to_plot)
    states_to_plot = states[nodes_to_plot_list]
    costs_to_plot = costs[nodes_to_plot_list]
    ani_ax.scatter(states_to_plot[:,0],states_to_plot[:,1],c=costs_to_plot,cmap=cmap)

    nodes_already_plotted = nodes_already_plotted.union(nodes_to_plot)


def update_frame(i): 
    #fade out
    ani_ax.clear()
    ani_ax.set_xlim(-1,1)
    ani_ax.set_ylim(-1,1)

    global nodes_already_plotted
    print 'frame: ',i
    t = plot_times[i]
    inds = np.where(states[:,2] <= t)[0]

    nodes_to_plot = set([nodes[i] for i in inds])
    #nodes_to_plot = nodes_to_plot.difference(nodes_already_plotted)

    ani_ax.set_title('time index: %d'%(t))
    nodes_to_plot_list = list(nodes_to_plot)
    states_to_plot = states[nodes_to_plot_list]
    costs_to_plot = costs[nodes_to_plot_list]
    for n in nodes_to_plot_list:
        age = t-states[n,2]
        alpha = .7**age
        if alpha>.01:
            ani_ax.plot(states[n,0],states[n,1],'o',mfc=cmap(costs[n]),alpha=alpha,zorder=int(age))

    nodes_already_plotted = nodes_already_plotted.union(nodes_to_plot)




ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=len(plot_times),interval=500)
ani.save('di_rrt_time_fade.mp4', fps=10, codec='mpeg4', clear_temp=True)

    
