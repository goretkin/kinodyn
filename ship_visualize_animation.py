from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:18:24 2012

@author: gustavo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:41:01 2012

@author: gustavo
"""

import shelve
import ipdb

from global_help import Rectangle_centered

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib as mpl
import numpy as np
import shapely.geometry as geo

def intersect_paths(paths1,paths2):
    for path1 in paths1:
        for path2 in paths2:
            if path1.intersects_path(path2):
                return True
    return False
    
class Ship_Sprite():

    def __init__(self):
        self.body_length = 3.0
        self.body_width = 2.0
        
        self.body_back_patch = Rectangle_centered((1.0,0),2.0,3.0,
                                                  linestyle='solid',linewidth=2)
        self.body_patch = Rectangle_centered((1.5,0.0),
                                                self.body_length,self.body_width,
                                                linestyle='solid',linewidth=1)
                
        self.lin_flame_patch = Rectangle_centered((0.0,0.0),1.0,1.0,linestyle='solid',color='g')
        self.ang_flame_patch = Rectangle_centered((2.0,0.0),1.0,1.0,linestyle='solid',color='r')

        self.patches = [self.body_patch,self.body_back_patch,self.lin_flame_patch,self.ang_flame_patch]
        
        self.body_patches = [self.body_back_patch,self.body_patch] #doesn't include flames
        #self.ship_collection = matplotlib.collections.PatchCollection(self.patches,match_original=True)
        
        #self.flame_color_map = matplotlib.cm.get_cmap(name='hot')
        self.flame_color_map = matplotlib.cm.get_cmap(name='autumn')
        
        #this is for optimization -- like representing the ship body as a single poly
        s = geo.Polygon()
        for p in self.body_patches:
            v = p.get_patch_transform().transform_path(p.get_path()).vertices
            s = s.union(geo.Polygon(v))
                    
        self.shapely_body = s.simplify(1e-6*self.body_length)
        self.ship_patch = mpl.patches.Polygon(np.array(self.shapely_body.exterior.xy).T)
        self.ship_exterior_path = self.ship_patch.get_path()
        
    def update_thrust(self,lin_thrust,ang_thrust):
        lin_flame_patch = self.lin_flame_patch
            
        if(lin_thrust == 0):
            lin_flame_patch.set_visible(False)
        else:
            lin_flame_patch.set_visible(True)
            
            lin_flame_width = .5
            lin_flame_size = abs(lin_thrust)
            
            lin_flame_patch.set_width(lin_flame_size)       #linear flame is rotated 
            lin_flame_patch.set_height(lin_flame_width)
            lin_flame_patch.set_color(self.flame_color_map(abs(lin_thrust)))
            
            #place linear flame in the right place
            if(lin_thrust > 0):
                lin_flame_patch.set_xy((-lin_flame_size/2,0.0))
            elif(lin_thrust < 0):
                lin_flame_patch.set_xy((3+lin_flame_size/2,0.0))        
        
        ang_flame_patch = self.ang_flame_patch
        
        if(ang_thrust == 0):
            ang_flame_patch.set_visible(False)
        else:
            ang_flame_patch.set_visible(True)
            
            ang_flame_width = .5
            ang_flame_size = abs(ang_thrust)
            ang_flame_patch.set_width(ang_flame_width)
            ang_flame_patch.set_height(ang_flame_size)
            ang_flame_patch.set_color(self.flame_color_map(abs(ang_thrust)))
        
            #put the angular flame in the right place    
            if(ang_thrust > 0):
                ang_flame_patch.set_xy(
                    (self.body_length-ang_flame_width ,
                     -ang_flame_size/2-self.body_width/2))
            else:        
                ang_flame_patch.set_xy(
                    (self.body_length-ang_flame_width ,
                     ang_flame_size/2+self.body_width/2))
                     
    def update_pose(self,x,y,theta):
        (self.x,self.y,self.theta) = (x,y,theta)
    
    def update_transform_axes(self,mpl_axes):
        trans0 = mpl_axes.transData
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans2 = matplotlib.transforms.Affine2D().rotate_around(0,0,self.theta)
        trans3 = matplotlib.transforms.Affine2D().scale(1)
        for p in self.patches:
            p.set_transform(trans3 + trans2+trans1+trans0)
            
    def collision(self,obstacle_paths):
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans2 = matplotlib.transforms.Affine2D().rotate_around(0,0,self.theta)
        
        trans = trans2+trans1
        
        return intersect_paths(obstacle_paths,
                               #[trans.transform_path(p) for p in mpl.collections.PatchCollection(self.patches).get_paths()]
                               [trans.transform_path(p.get_path()) for p in self.body_patches]
                               )
    def collision1(self,obstacle_patches):
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans2 = matplotlib.transforms.Affine2D().rotate_around(0,0,self.theta)
        
        trans = trans2+trans1
        
        transformed_obstacle_paths = [p.get_patch_transform().transform_path(p.get_path()) for p in obstacle_patches]
        return intersect_paths(transformed_obstacle_paths,
                               #[trans.transform_path(p) for p in mpl.collections.PatchCollection(self.patches).get_paths()]
                               [trans.transform_path(p.get_path()) for p in self.body_patches]
                               )
    def collision2(self,obstacle_paths):
        #use a single ship patch
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans2 = matplotlib.transforms.Affine2D().rotate_around(0,0,self.theta)
        
        trans = trans2+trans1
        
        return intersect_paths(obstacle_paths,
                               [trans.transform_path(self.ship_patch.get_path())]
                               )
                               
    def collision3(self,shapely_multipoly):
        #broken
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans2 = matplotlib.transforms.Affine2D().rotate_around(0,0,self.theta)        
        trans = trans2+trans1
        
        xy = trans.transform_path(self.ship_exterior_path).vertices
        return shapely_multipoly.intersects(geo.Polygon(xy))
        
    def set_alpha(self,alpha):
        for p in self.patches:
            p.set_alpha(alpha)


def benchmark_collision3(n):
    a = Ship_Sprite()
    for i in np.linspace(0,20,n):
        theta = np.random.random()*np.pi*2
        a.update_pose(i,0,theta)    
        a.collision3(shapely_obstacles)
        
def benchmark_collision2(n):
    a = Ship_Sprite()
    for i in np.linspace(0,20,n):
        theta = np.random.random()*np.pi*2
        a.update_pose(i,0,theta)    
        a.collision2(obstacle_pc.get_paths())
        
def benchmark_collision(n):
    a = Ship_Sprite()
    for i in np.linspace(0,20,n):
        theta = np.random.random()*np.pi*2
        a.update_pose(i,0,theta)
        a.collision(obstacle_pc.get_paths())        
        
        
if __name__ == '__main__':
    ship_shelve = shelve.open('ship.shelve')
    field_shelve = shelve.open('field.shelve')
    
    obstacle_paths = field_shelve['obstacle_paths']
    
    print 'loading local vars',ship_shelve.keys()
    
    #to pacify the linter
    traj = None
    utraj = None
    T = None
    
    for k in ship_shelve.keys():
        locals()[k]=ship_shelve[k]
        
    shapely_obstacles = geo.Polygon()
    pps = []
    for op in obstacle_paths:
        trans = mpl.transforms.Affine2D().scale(15)+mpl.transforms.Affine2D().translate(35,20)
        op = trans.transform_path(op)
        pp = matplotlib.patches.PathPatch(path=op,color='k')
        pps.append(pp)
        shapely_obstacles = shapely_obstacles.union(geo.Polygon(op.vertices))
        #trail_plot.add_patch(pp)
    obstacle_pc = matplotlib.collections.PatchCollection(pps,match_original=True)
    
    if False:
        path_figure = plt.figure(None)
        path_figure.gca().plot(traj[:,3],traj[:,4],'.')
        path_figure.gca().set_aspect('equal')
        
    trail_figure = plt.figure(None)
    
    trail_plot = trail_figure.gca() #.subplot(111,aspect='equal')
    trail_plot.set_axis_bgcolor((.9,.9,.9)) #gray background
    
    trail_indices = np.arange(0,T,20)
    trail_traj = traj[trail_indices,:] #samples of trajectory to draw
    
    max_lin_thrust = np.max(np.abs(utraj[:,0]))
    max_ang_thrust = np.max(np.abs(utraj[:,1]))

    for i in trail_indices:
        a = Ship_Sprite()
        a.update_thrust(lin_thrust=utraj[i,0]/max_lin_thrust,
                        ang_thrust=utraj[i,1]/max_ang_thrust)
        a.update_pose(traj[i,3],traj[i,4],traj[i,5])
        a.update_transform_axes(trail_plot)
    
        a.set_alpha(0.3)
        collide = a.collision2(obstacle_pc.get_paths())
        if collide:
            print i,'collide'
        for p in a.patches:
            #p.set_alpha(0.6)
            if(collide):
                p.set_color('r')
            trail_plot.add_artist(p)                    
        
    trail_plot.set_xlim(np.min(traj[:,3])-10,np.max(traj[:,3])+10)
    trail_plot.set_ylim(np.min(traj[:,4]-10),np.max(traj[:,4])+10)
    trail_plot.set_aspect('equal')
    
    trail_plot.add_collection(obstacle_pc)

    assert False    
    ani_fig = plt.figure(None)
    ani_ax = ani_fig.gca()
    
    ani_ax.set_xlim(np.min(traj[:,3])-10,np.max(traj[:,3])+10)
    ani_ax.set_ylim(np.min(traj[:,4]-10),np.max(traj[:,4])+10)
    ani_ax.set_aspect('equal')
    
    ship_sprite = Ship_Sprite()
    
    for p in ship_sprite.patches:
        ani_ax.add_artist(p)
    
    #import copy
    def update_frame(i):
        j = trail_indices[i]
        ship_sprite.update_pose(traj[j,3],traj[j,4],traj[j,5])
        ship_sprite.update_thrust(utraj[j,0]/max_lin_thrust,
                                  utraj[j,1]/max_ang_thrust)
        ship_sprite.update_transform_axes(ani_ax)
        ani_ax.set_title('time index: %d'%(j))
        #ani_ax.add_artist(copy.copy(ship_sprite.ship_collection))
     
    ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=trail_indices.size,interval=50)
    #ani.save('test.mp4', fps=20, codec='mpeg4', clear_temp=True)
