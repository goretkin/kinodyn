import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import networkx as nx


class RRT_Interactive():
    def __init__(self,rrt,run_forward=None,cost_traj=None,plot_dims=[0,1],slider_dim=-1,slider_range=(0,100)):
        self.rrt = rrt
        self.run_forward = run_forward
        self.cost_traj = cost_traj
        self.plot_dims = plot_dims
        self.slider_dim = self.rrt.state_ndim-1 if slider_dim==-1 else slider_dim

        self.int_fig = plt.figure(None)
        self.int_ax = self.int_fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=.15,bottom=.35,right=.8)
        self.action_ts_ax = plt.axes([.15,.2,.65,.1])

        self.info_text = None #text object plotting number of nodes.
        self.to_clear = []  #things that need to be cleared when plotting
        
        self.enable_force_extension_from = True #right click to select a node to extend from
        self.extend_from_node = None

        if(self.slider_dim is not None):
            self.time_slider_ax = plt.axes([.15,.1,.65,.03])        
            self.time_slider = mpl.widgets.Slider(self.time_slider_ax,'Time',slider_range[0],slider_range[1],valinit=1)
    
        # shift the axis to make room for legend
        #box = int_ax.get_position()
        #int_ax.set_position([box.x0-.1, box.y0+.15, box.width, box.height])
        #plt.tight_layout()
        self.draw_rrt(self.int_ax,self.action_ts_ax)
        self.sample_template = np.zeros(self.rrt.state_ndim)

        def button_press_event_dispatcher(event):
            if self.int_fig.canvas.widgetlock.locked(): #matplotlib widget in use
                return

            if event.inaxes is not self.int_ax: #did not click in axes
                return

            if event.button == 1: #dominant click, sample this point 
                sample = np.array(self.sample_template)
                sample[self.plot_dims] = (event.xdata,event.ydata)
                if self.slider_dim is not None:
                    sample[slider_dim] = int(self.time_slider.val)   #discrete time
                print 'sample',sample

                self.rrts(sample)         

            elif event.button == 3: #print node info on right click.
                #node_id, distance = interactive_rrt.nearest_neighbor([event.xdata,event.ydata,interactive_T])
                nodes = self.rrt.tree.nodes()
                pos = np.array([self.rrt.tree.node[i]['state'][self.plot_dims] for i in nodes])
                distances = np.sum( (pos - np.array([event.xdata,event.ydata]))**2,axis=1)
                closest = np.argmin(distances)
                node_id = nodes[closest]
                
                state = self.rrt.tree.node[node_id]['state']
                self.int_ax.text(state[self.plot_dims[0]],state[self.plot_dims[1]],s=str(node_id),zorder=30)    #text on top
                self.int_fig.canvas.draw()
                
                print node_id, self.rrt.tree.node[node_id].keys()
                if self.enable_force_extension_from:
                    self.extend_from_node = node_id
                
            import sys
            sys.stdout.flush() #function is called asyncrhonously, so any print statements might not flush

        self.int_fig.canvas.mpl_connect('button_press_event', button_press_event_dispatcher)            

    def rrts(self,xrand=None):
        if xrand is not None:
            xrand = np.array(xrand)
            if self.extend_from_node is not None:
                print 'extending forced from node',self.extend_from_node
                self.rrt.extend_from(self.extend_from_node,xrand)
                #self.extend_from_node = None
            else:
                self.rrt.extend(xrand)
        print 'start draw'
        self.draw_rrt(self.int_ax,self.action_ts_ax)
        print 'end draw'
        self.rrt.viz_change = False
        print 'start update canvas'
        self.int_fig.canvas.draw()
        print 'done update canvas'


    def draw_rrt(self,int_ax,control_ax=None,cost_coded_edges=False, colormap = None, hide_legend=True):
        ax = int_ax
        if colormap is None:
            colormap = mpl.cm.get_cmap(name='copper')

        for thing in self.to_clear:
            if type(thing) in [matplotlib.collections.LineCollection, matplotlib.collections.PatchCollection, matplotlib.collections.Collection]:
                thing.remove()
            if type(thing) == matplotlib.lines.Line2D:
                ax.lines.remove(thing)
            
        self.to_clear = []

        #ax.cla()
        #ax.set_xlim(-1,1)
        #ax.set_ylim(-1.5,1)

        #for l in ax.lines:
        #    l.remove()
        #for p in ax.patches:
        #    p.remove()
            
        tree = self.rrt.tree
        rrt = self.rrt

        if (rrt.viz_change):            
            viz_x_nearest = rrt.viz_x_nearest
            viz_x_new = rrt.viz_x_new
            viz_x_from =rrt.viz_x_from
        
        """        
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
        """
        #draw best solution
        if len(rrt.cost_history) > 0:
            x0 = rrt.state0
            utraj = rrt.cost_history[-1][2][2]
            xs = self.run_forward(x0,utraj)
            xs = np.concatenate((x0.reshape((1,-1)),xs))
            xs = xs[:,self.plot_dims]
            ax.plot(xs[:,0],xs[:,1],'-',lw=3,alpha=.9,color='red',label='best solution',zorder=30)

        #draw paths that collided
        if (not rrt.viz_collided_paths is None) and (not self.run_forward is None):
            if len(rrt.viz_collided_paths) > 100:
                #too many collided paths, don't bother drawing.
                print 'not drawing collided paths. too many ({})'.format(len(rrt.viz_collided_paths))
            else:
                lines = []
                for (node,action) in rrt.viz_collided_paths:
                    x0 = node['state']
                    xs = self.run_forward(x0,action)
                    xs = np.concatenate((x0.reshape((1,-1)),xs))
                    lines.append(xs[:,self.plot_dims])
                
                collision_collection = mpl.collections.LineCollection(lines,linewidths=1,linestyles='solid')
                collision_collection.set_color('red')
                collision_collection.set_alpha(.5)
                collision_collection.set_zorder(1)
                collision_collection_added = ax.add_collection(collision_collection)
                self.to_clear.append(collision_collection_added)

            rrt.viz_collided_paths = [] #reset in order to plot only new collided paths

        if (rrt.viz_change):
            #draws a straight edge
            new_ext_x = [viz_x_from[self.plot_dims[0]],viz_x_new[self.plot_dims[0]]]
            new_ext_y = [viz_x_from[self.plot_dims[1]],viz_x_new[self.plot_dims[1]]]
            ax.plot(new_ext_x,new_ext_y,'y',lw=5,alpha=.7,zorder=3,label='new extension')
            

        pos = {n:tree.node[n]['state'][self.plot_dims] for n in tree.nodes()}
        col = [tree.node[n]['cost'] for n in tree.nodes()]

        ax.get_figure().sca(ax) #set the current axis to the int_ax. there is some bug in networkx/matplotlib
        node_collection = nx.draw_networkx_nodes(G=tree,
                                                pos=pos,
                                                ax=ax,
                                                node_size=1 if cost_coded_edges else 15,
                                                node_color='black' if cost_coded_edges else col,
                                                cmap = None if cost_coded_edges else colormap,
                                                node_shape = '.' if cost_coded_edges else 'o'   #if color is in the edges, don't need big nodes
                                                )

        if not node_collection is None:
            node_collection.set_zorder(5)
            self.to_clear.append(node_collection)

        if self.run_forward is None:                                        
            #draw straight edges
            edge_collection = nx.draw_networkx_edges(G=tree,
                                                    pos=pos,
                                                    ax=ax,
                                                    edge_color='b',
                                                    )
        else:
            if not cost_coded_edges:
                #draw uniform color dynamical edges
                if not self.__dict__.has_key('edge_cache'):
                    'reset edge_cache'
                    self.edge_cache = {}

                lines = []
                for i in tree.nodes():
                    if self.edge_cache.has_key(i):
                        xs = self.edge_cache[i]
                    else:
                        s = tree.predecessors(i)
                        if len(s) == 0:
                            continue
                        assert len(s) == 1 #it's a tree
                        s = s[0]
                        x0 = tree.node[s]['state']
                        xs = self.run_forward(x0, tree.node[i]['action'])
                        xs = np.concatenate((x0.reshape((1,-1)),xs))
                        self.edge_cache[i] = xs

                    lines.append(xs[:,self.plot_dims])
                    
                edge_collection = mpl.collections.LineCollection(lines)
                ax.add_collection(edge_collection)

            else:
                #shade edges with cost
                if self.cost_traj is None: raise ValueError("Color-coded edges requires a cost function")
                lines = []
                costs = []
                for i in tree.nodes():
                    s = tree.predecessors(i)
                    if len(s) == 0:
                        continue
                    assert len(s) == 1 #it's a tree
                    s = s[0]
                    x0 = tree.node[s]['state']
                    c0 = tree.node[s]['cost']
                    utraj = tree.node[i]['action']
                    
                    for j in range(len(utraj)):
                        u = utraj[[j]]  #chop up the action
                        xs = self.run_forward(x0,u)
                        assert len(xs) == 1 #one-step trajectory
                        xs = xs[0]
                        c0 += self.cost_traj(x0,u)
                        #the edge  between x0 and xs has cost c0
                        line = [ x0[self.plot_dims], xs[self.plot_dims] ] 
                        lines.append(line)
                        costs.append(c0)
                        
                        x0 = xs
                
                edge_collection = mpl.collections.LineCollection(lines,cmap=colormap)
                costs = np.array(costs).reshape((-1,))
                edge_collection.set_array(costs) #shade the edges
                edge_collection.set_alpha(.6)
                ax.add_collection(edge_collection)
                
                    
        if not edge_collection is None:
                edge_collection.set_zorder(4)
                self.to_clear.append(node_collection)       
        
        #mfc, mec, mew is marker face color, edge color, edge width
        if (rrt.viz_change):

            #ani_ax.add_patch(mpl.patches.Circle(xy=viz_x_new,radius=ani_rrt.viz_search_radius,
            #                                    alpha=.3,fc='none',ec='b',label='_rewire radius'))
        
            self.to_clear.extend(   ax.plot(*rrt.viz_x_rand[self.plot_dims],marker='*', mfc='k', mec='k', ls='None', zorder=6, label='x_rand')          )
            self.to_clear.extend(   ax.plot(*viz_x_nearest[self.plot_dims],marker='p', mfc='c', mec='c', ls='None', zorder=7, ms=5, label='x_nearest')  )
            self.to_clear.extend(   ax.plot(*viz_x_new[self.plot_dims], marker='x', mfc='r', mec='r', ls='None', zorder=8, label='x_new')               )
            self.to_clear.extend(   ax.plot(*viz_x_from[self.plot_dims], marker='o', mfc='g',mec='g', ls='None',alpha=.5, zorder=9, label='x_from')     )
            
            if rrt.viz_x_near is not None and len(rrt.viz_x_near)>0:
                x_near = np.array(rrt.viz_x_near)
                self.to_clear.extend(   ax.plot(x_near[:,self.plot_dims[0]],x_near[:,self.plot_dims[1]], marker='o', mfc='none',mec='r', mew=1 ,ls='None',alpha=.5, zorder=10, label='X_near')  )
    
        if not hide_legend:
            ax.legend(bbox_to_anchor=(1.05,0.0),loc=3,
                           ncol=1, borderaxespad=0.,
                            fancybox=True, shadow=True,numpoints=1)
            
            #ani_ax.legend()
            if ax.get_legend() is not None:
                self.to_clear.append(ax.get_legend())
                plt.setp(ax.get_legend().get_texts(),fontsize='small')

            info = ""
            info += "# nodes: %d\n" % (len(tree.nodes()))
            #info += "# edges: %d\n" % (len(tree.edges()))
            info += "cost: %s\n" % (str(rrt.worst_cost) if rrt.found_feasible_solution else "none")

            if self.info_text is None:
                self.info_text = ax.figure.text(.8, .5, info,size='small')
            self.info_text.set_text(info)


