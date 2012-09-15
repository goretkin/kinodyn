
from joblib import Parallel, delayed
import examplets_import
#import rrt_2d_example as rrt_setup
import linear_ship_rrt as rrt_setup

import shelve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import copy
import os

a = []
n_iters_s = []

folder = '/home/goretkin/diffruns'

def load_all_shelve(folder):
    s = []
    sname = []
    for files in os.listdir(folder):
        if files.endswith(".shelve"):
            s.append(shelve.open(os.path.join(folder,files)))
            sname.append(os.path.join(folder,files))
    for files in os.listdir(folder):
        if files.endswith(".shelve.temp") and files[:-5] not in sname:  #don't add a tempfile for the same run
            s.append(shelve.open(os.path.join(folder,files)))
            print 'adding temp file: {}'.format(files)
    return s
  

shelves = load_all_shelve(folder)
for s in shelves:       
    if s.has_key('cost_history'):
        a.append(s['cost_history'])
        n_iters_s.append(s['n_iters'])
    else:
        print '{} missing.'.format(files)
    #s.close()

n_rrts = len(a) #number of runs of RRT we have.
        
cost_histories = []

n_iters = 5000

for i in range(len(a)):
    if len(a[i]) == 0: 
        print 'RRT {} did not solve'.format(i)
        continue  #did not find first solution
    first_solution_itr = a[i][0][0]     #the first iteration at which a solution was found. make zero in order to not align to first solution
    
    print 'Iterations after first solve: {}'.format(n_iters_s[i] - first_solution_itr)

    cost_history = np.zeros(n_iters)
    for j in range(len(a[i])):
        k = a[i][j][0]-first_solution_itr   #iteration relative to first solution
        if k >= n_iters: break
        cost_history[k:] = a[i][j][1]
    cost_histories.append(cost_history)        

average_first_solve = np.mean( [a[i][0][0] for i in range(len(a))] )

cost_histories = np.array(cost_histories).T
#cost_histories_masked = np.ma.masked_where(cost_histories == np.inf, cost_histories)

cost_history_avg = np.mean(cost_histories,axis=1)
cost_history_std = np.std(cost_histories,axis=1)
err_pos_x = np.arange(0,len(cost_history_avg),50)

x_offset = 0 #int(average_first_solve)
pos_x = np.arange(0,len(cost_history_avg)) + x_offset
ax = plt.figure().gca()
ax.plot(pos_x, cost_history_avg)
#ax.semilogy(cost_history_avg)
ax.errorbar(x=err_pos_x+x_offset ,y=cost_history_avg[err_pos_x],yerr=cost_history_std[err_pos_x],fmt=None) #only add error bars (fmt=None)

ax.set_title('Convergence plot ({} runs)'.format(n_rrts))
ax.set_xlabel('Iterations after first solution ({0:.1})'.format(average_first_solve))
ax.set_ylabel('Cost')

ax = plt.figure().gca()
ax.plot(cost_histories)

def run_forward(start,action):
    return np.cumsum(action,axis=0) + start


run_forward = rrt_setup.lqr_rrt.run_forward

cost_min = np.min(cost_histories)
cost_max = np.max(cost_histories)


colormap = matplotlib.cm.get_cmap(name='copper')
solution_costs = []

solutions = []

plot_all = True

if plot_all:
    ax = plt.figure().gca()
for i in range(len(a)):
    print 'processing {} of {}'.format(i,len(a))
    for j in range(len(a[i])):
        #i ranges over different runs of RRT and j ranges over different solution improvements in i.
        iteration = a[i][j][0]
        cost = a[i][j][1]
        solution_costs.append(cost)
        best_solution = a[i][j][2]
        u_path = best_solution[2]
        start_state = best_solution[1][0]
        x_path = run_forward(start_state,u_path)
        solutions.append( (cost,x_path,u_path) )
        if plot_all:
            normalized_cost = (cost-cost_min)/(cost_max-cost_min)
            ax.plot(x_path[:,2],x_path[:,3],alpha=.2,color=colormap(np.log(normalized_cost+1))[0:3],lw=3)

from ship_field import get_patch_collection
ax.add_collection(get_patch_collection())

ax.set_title('Solutions color-coded by cost')

solutions.sort(key=lambda x: x[0])
solutions = solutions[::-1] #most expensive first

best_cost,best_x_traj,best_u_traj = solutions[-1]

best_shelve = shelve.open('rrt_best.shelve')
best_shelve['traj'] = best_x_traj
best_shelve['utraj'] = best_x_traj
best_shelve.close()
print 'made best shelve'

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







