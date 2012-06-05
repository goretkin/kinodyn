# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:41:01 2012

@author: gustavo
"""

import shelve

ship_shelve = shelve.open('ship.shelve')
print 'loading local vars',ship_shelve.keys()

#to pacify the linter
traj = None
utraj = None
T = None

for k in ship_shelve.keys():
    locals()[k]=ship_shelve[k]


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np

path_figure = plt.figure(None)
path_figure.gca().plot(traj[:,3],traj[:,4],'.')
path_figure.gca().set_aspect('equal')
    
trail_figure = plt.figure(None)

trail_plot = trail_figure.gca() #.subplot(111,aspect='equal')
trail_plot.set_axis_bgcolor((.9,.9,.9)) #gray background

trail_indices = np.arange(0,T,10)
trail_traj = traj[trail_indices,:] #samples of trajectory to draw


def Rectangle_centered(xy,width,height,*args,**kwargs):
    x = xy[0] - width/2
    y = xy[1] - height/2
    rect = matplotlib.patches.Rectangle((x,y),width,height,*args,**kwargs)
    rect.set_xy = def set_xy(self,xy):
        self._x = xy[0] - self._width/2
        self._y = xy[1] - self._height/2
    return rect
    

max_lin_thrust = np.max(np.abs(utraj[:,0]))
max_ang_thrust = np.max(np.abs(utraj[:,1]))

flame_color_map = matplotlib.cm.get_cmap(name='hot')
flame_color_map = matplotlib.cm.get_cmap(name='autumn')

def ship_patch(lin_thrust,ang_thrust):
    body_length = 3.0
    body_width = 2.0
    #the thrust is between -1 and 1
    patches = [Rectangle_centered((1.0,0),2.0,3.0,linestyle='solid',linewidth=2),
               Rectangle_centered((1.5,0.0),body_length,body_width,linestyle='solid',linewidth=1),
            ]
    if(lin_thrust > 0):
        flame_size = lin_thrust+0.05
        patches.append(Rectangle_centered((-flame_size/2,0),flame_size,.5,linestyle='solid',
                                          color=flame_color_map(lin_thrust)))
    elif(lin_thrust < 0):
        flame_size = -lin_thrust+0.05
        patches.append(Rectangle_centered((3+flame_size/2,0),flame_size,.5,linestyle='solid',
                                          color=flame_color_map(-lin_thrust)))
    
    ang_flame_width = .5
    if(ang_thrust > 0):
        flame_size = ang_thrust+0.05
        patches.append(Rectangle_centered((body_length-ang_flame_width ,-flame_size/2-body_width/2),
                                          ang_flame_width ,flame_size,linestyle='solid',
                                          color=flame_color_map(ang_thrust)))
    elif(ang_thrust < 0):
        flame_size = -ang_thrust+0.05
        patches.append(Rectangle_centered((body_length-ang_flame_width ,flame_size/2+body_width/2),
                                          ang_flame_width ,flame_size,linestyle='solid',
                                          color=flame_color_map(-ang_thrust)))                                          
                                          
    return matplotlib.collections.PatchCollection(patches,match_original=True)

ship_patches = [ship_patch(lin_thrust=utraj[x,0]/max_lin_thrust,
                           ang_thrust=utraj[x,1]/max_ang_thrust) for x in trail_indices]
                

for i in range(len(ship_patches)):
    x = trail_traj[i,:]
    #trans0 = ship_patches[i].get_transform()
    trans0 = trail_plot.transData
    trans1 = matplotlib.transforms.Affine2D().rotate_around(x[3],x[4],x[5])
    trans2 = matplotlib.transforms.Affine2D().translate(x[3],x[4])
    trans3 = matplotlib.transforms.Affine2D().scale(4)
    ship_patches[i].set_transform(trans3 + trans2+trans1+trans0)


for e in ship_patches:
    #e.set_clip_box(trail_plot.bbox)
    e.set_alpha(0.9)
    trail_plot.add_artist(e)
    
trail_plot.set_xlim(np.min(traj[:,3])-10,np.max(traj[:,3])+10)
trail_plot.set_ylim(np.min(traj[:,4]-10),np.max(traj[:,4])+10)
trail_plot.set_aspect('equal')


