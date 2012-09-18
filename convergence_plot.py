
from joblib import Parallel, delayed
import examplets_import
#import rrt_2d_example as rrt_setup
#import linear_ship_rrt as rrt_setup
#import pendulum as rrt_setup
import linear_ship_rrt_cost as rrt_setup

import shelve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import copy
import os
import sys

a = []
n_iters_s = []
rrt_shelves = []

folder = '/home/goretkin/diffruns'

if len(sys.argv) > 1:
    folder = sys.argv[1]

print 'using folder: {}'.format(folder)

def load_all_shelve(folder):
    s = []
    sname = []
    f = []
    for files in os.listdir(folder):
        if files.endswith(".shelve"):
            s.append(shelve.open(os.path.join(folder,files),flag='r'))
            sname.append(files)
            f.append(files)
    for files in os.listdir(folder):
        if files.endswith(".shelve.temp") and files[:-5] not in sname:  #don't add a tempfile for the same run
            s.append(shelve.open(os.path.join(folder,files),flag='r'))
            print 'adding temp file: {}'.format(files)
            f.append(files)
    return s, f

shelves,files = load_all_shelve(folder)

for (s,f) in zip(shelves,files):       
    if s.has_key('cost_history'):
        a.append(s['cost_history'])
        n_iters_s.append(s['n_iters'])
        rrt_shelves.append(s)
    else:
        print '{} missing.'.format(f)
    #s.close()

n_rrts = len(rrt_shelves) #number of runs of RRT we have.
        
cost_histories = []

n_iters = 4000

for i in range(n_rrts):
    a = rrt_shelves[i]['cost_history']
    if len(a) == 0: 
        print 'RRT {} did not solve'.format(i)
        continue  #did not find first solution
    first_solution_itr = a[0][0]     #the first iteration at which a solution was found. make zero in order to not align to first solution
    
    iters_after_first_sol = n_iters_s[i] - first_solution_itr
    if iters_after_first_sol < n_iters :
        print 'Iterations after first solve only: {}!'.format(n_iters_s[i] - first_solution_itr)
    
    
    cost_history = np.zeros(n_iters)
    for j in range(len(a)):
        k = a[j][0]-first_solution_itr   #iteration relative to first solution
        if k >= n_iters: break
        cost_history[k:] = a[j][1]
    cost_histories.append(cost_history)        

average_first_solve = np.mean( [a[0][0] for i in range(len(a))] )

cost_histories = np.array(cost_histories).T
#cost_histories_masked = np.ma.masked_where(cost_histories == np.inf, cost_histories)

cost_history_avg = np.mean(cost_histories,axis=1)
cost_history_std = np.std(cost_histories,axis=1)/np.sqrt(n_rrts)
err_pos_x = np.arange(0,len(cost_history_avg),50)

x_offset = 0 #int(average_first_solve)
pos_x = np.arange(0,len(cost_history_avg)) + x_offset
ax = plt.figure().gca()
ax.plot(pos_x, cost_history_avg)
#ax.semilogy(cost_history_avg)
ax.errorbar(x=err_pos_x+x_offset ,y=cost_history_avg[err_pos_x],yerr=cost_history_std[err_pos_x],fmt=None) #only add error bars (fmt=None)

ax.set_title('Convergence plot ({} runs)'.format(n_rrts))
ax.set_xlabel('Iterations after first solution ({0:0.2})'.format(average_first_solve))
ax.set_ylabel('Cost')

ax = plt.figure().gca()
ax.plot(cost_histories)

def run_forward(start,action):
    return np.cumsum(action,axis=0) + start


run_forward = rrt_setup.lqr_rrt.run_forward

cost_min = np.min(cost_histories)
cost_max = np.max(cost_histories)



solution_costs = []

solutions = []

plot_all = True

plot_dims = [2,3]

for i in range(n_rrts):
    a = rrt_shelves[i]['cost_history']
    print 'processing {} of {}'.format(i,n_rrts)
    for j in range(len(a)):
        #i ranges over different runs of RRT and j ranges over different solution improvements in i.
        iteration = a[j][0]
        cost = a[j][1]
        solution_costs.append(cost)
        best_solution = a[j][2]
        u_path = best_solution[2]
        start_state = best_solution[1][0]
        x_path = run_forward(start_state,u_path)
        solutions.append( (cost,x_path,u_path) )

solutions.sort(key=lambda x: x[0])  #sort by cost
solutions = solutions[::-1] #most expensive first

n = len(solutions)
solutions_to_plot = solutions[0:n/4] + solutions[:n/4:10]
colormap = matplotlib.cm.get_cmap(name='copper')
if plot_all:
    ax = plt.figure().gca()
    for (i,solution) in enumerate(solutions_to_plot):
        (cost,x_path,u_path) = solution
        normalized_cost = (cost-cost_min)/(cost_max-cost_min)
        ax.plot(x_path[:,plot_dims[0]],x_path[:,plot_dims[1]],alpha=.3,color=colormap(np.log(normalized_cost+1))[0:3],lw=2,zorder=i)

from ship_field import get_patch_collection
pc = get_patch_collection()
pc.set_color('gray')
ax.add_collection(pc)
ax.set_xlim(-10,110)
ax.set_ylim(-10,110)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_aspect('equal')
ax.set_title('All solutions found over {} runs.'.format(n_rrts))



best_cost,best_x_traj,best_u_traj = solutions[-1]

best_shelve = shelve.open('rrt_pendulum_best.shelve')
best_shelve['traj'] = best_x_traj
best_shelve['utraj'] = best_u_traj
best_shelve.close()
print 'made best shelve'

plt.show()
if False:
    import matplotlib.animation as animation
    ani_fig = plt.figure()
    ani_ax = ani_fig.gca()

    last_line = []
    def update_frame(i):
        global last_line
        for line in last_line: ani_ax.lines.remove(line)
        cost,x_path = solutions[i]
        print i,cost
        ani_ax.set_title('cost: {}'.format(cost))
        normalized_cost = (cost-cost_min)/(cost_max-cost_min)
        last_line = ani_ax.plot(x_path[:,2],x_path[:,3],color=colormap(normalized_cost))

     
    ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=len(solutions),interval=50)
    ani.save('solution.mp4', fps=20, codec='mpeg4', clear_temp=True)







