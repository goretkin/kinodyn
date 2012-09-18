import pendulum
import shelve
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np


if False: #load an rrt
    s = shelve.open('/home/goretkin/diffruns/rrt_0000.shelve')

    rrt = pendulum.rrt

    rrt.load(s)

    utraj = rrt.cost_history[-1][2][2]
    xtraj = pendulum.lqr_rrt.run_forward(rrt.state0,utraj)
else:   #load the trajectory only
    s = shelve.open('/home/goretkin/Dropbox/kinodyn/rrt_pendulum_best.shelve')

    utraj = s['utraj']
    xtraj = s['traj']

t = len(utraj)
t_subsample = range(0,t,5)

length = 1 #length of pendulum
trail_ax = plt.figure(None).gca()
trail_ax.set_ylim(-1.5,1.5)
trail_ax.set_xlim(-2,2)

cmap = mpl.cm.hsv
for i in range(len(t_subsample)):
    _t = t_subsample[i]
    pendulum_body = trail_ax.plot([0,0],[0,length])[0]

    x = -length*np.cos(xtraj[_t,1]-np.pi/2)
    y = length*np.sin(xtraj[_t,1]-np.pi/2)

    pendulum_bob = mpl.patches.Circle(xy=(x,y),radius=.05)

    pendulum_body.set_data( [0, x],
                            [0, y]
                            )
    pendulum_body.set_linewidth(2)
    pendulum_body.set_alpha(.7)    
    pendulum_body.set_color(cmap(float(_t)/t))
    pendulum_bob.set_alpha(1)
    trail_ax.add_patch(pendulum_bob)
    
    #plot later configurations on top
    pendulum_body.set_zorder(2*i)
    pendulum_bob.set_zorder(2*i+1)  
    



ani_fig = plt.figure(None)
x_ax = ani_fig.add_subplot(2,1,1)
u_ax = ani_fig.add_subplot(2,1,2)


u_ax.plot(utraj)
time_line = u_ax.axvline(x=0)

length = 1
pendulum_body = x_ax.plot([0,0],[0,length])[0]

x_ax.set_ylim(-1.5,1.5)
x_ax.set_xlim(-2,2)
x_ax.set_aspect('equal')

def update_frame(i):
    t = t_subsample[i]
    print t
    time_line.set_data( ([t,t],[0,1]) )  #scan time

    x_ax.set_title('time index: %d'%(t))
#    x_ax.plot(xtraj[0:t,0],xtraj[0:t,1])

    pendulum_body.set_data( [0, -length*np.cos(xtraj[t,1]-np.pi/2)],
                            [0, length*np.sin(xtraj[t,1]-np.pi/2)]
                            )

    #ani_ax.add_artist(copy.copy(ship_sprite.ship_collection))
     
ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=len(t_subsample),interval=50)
#ani.save('pendulum_swingup.mp4', fps=20, codec='mpeg4', clear_temp=True)
