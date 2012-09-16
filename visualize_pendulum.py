import pendulum
import shelve
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np


s = shelve.open('/home/goretkin/diffruns/rrt_0000.shelve')

rrt = pendulum.rrt

rrt.load(s)

utraj = rrt.cost_history[-1][2][2]
xtraj = pendulum.lqr_rrt.run_forward(rrt.state0,utraj)


ani_fig = plt.figure(None)
x_ax = ani_fig.add_subplot(2,1,1)
u_ax = ani_fig.add_subplot(2,1,2)
t = len(utraj)
t_subsample = range(0,t,10)

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
ani.save('pendulum_swingup.mp4', fps=20, codec='mpeg4', clear_temp=True)
