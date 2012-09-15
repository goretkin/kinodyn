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

from global_help import Rectangle_centered

import matplotlib.patches
import matplotlib.pyplot as plt
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
        self.body_length = 3.0  #vertical width
        self.body_width = 3.0   #horizontal height
                        
        self.horz_flame_patch = Rectangle_centered((0.0,0.0),1.0,1.0,linestyle='solid',color='g')
        self.vert_flame_patch = Rectangle_centered((2.0,0.0),1.0,1.0,linestyle='solid',color='r')
        self.body_patch = matplotlib.patches.Circle(xy=(0,0),radius=1.5)

        self.patches = [self.body_patch,self.horz_flame_patch,self.vert_flame_patch]
        self.body_patches = [self.body_patch]

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
        
    def update_thrust(self,horz_thrust,vert_thrust):
        horz_flame_patch = self.horz_flame_patch
            
        if(horz_thrust == 0):
            horz_flame_patch.set_visible(False)
        else:
            horz_flame_patch.set_visible(True)
            
            horz_flame_width = .5 + (abs(horz_thrust)**4)/2
            horz_flame_size = 2*abs(horz_thrust)
            
            horz_flame_patch.set_width(horz_flame_size)
            horz_flame_patch.set_height(horz_flame_width)
            horz_flame_patch.set_color(self.flame_color_map(abs(horz_thrust)))
            
            #place horz flame in the right place
            if(horz_thrust > 0):
                horz_flame_patch.set_xy((-self.body_length/2 -horz_flame_size/2,0.0))
            elif(horz_thrust < 0):
                horz_flame_patch.set_xy((self.body_length/2 + horz_flame_size/2,0.0))        
        
        vert_flame_patch = self.vert_flame_patch
        
        if(vert_thrust == 0):
            vert_flame_patch.set_visible(False)
        else:
            vert_flame_patch.set_visible(True)
            
            vert_flame_width = .5 + (abs(vert_thrust)**4)/2
            vert_flame_size = 2*abs(vert_thrust)
            vert_flame_patch.set_width(vert_flame_width)
            vert_flame_patch.set_height(vert_flame_size)
            vert_flame_patch.set_color(self.flame_color_map(abs(vert_thrust)))
        
            #put the angular flame in the right place    
            if(vert_thrust > 0):
                vert_flame_patch.set_xy(
                    (0 ,
                     -vert_flame_size/2-self.body_width/2))
            else:        
                vert_flame_patch.set_xy(
                    (0 ,
                     vert_flame_size/2+self.body_width/2))
                     
    def update_pose(self,x,y):
        (self.x,self.y) = (x,y)
    
    def update_transform_axes(self,mpl_axes):
        trans0 = mpl_axes.transData
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans3 = matplotlib.transforms.Affine2D().scale(1)
        for p in self.patches:
            p.set_transform(trans3 + trans1+trans0)
  
    def get_ship_path(self):
        """
        get transformed path
        """
        trans = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        return trans.transform_path(self.ship_patch.get_path())
          
    def collision(self,obstacle_paths):
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        
        trans = trans1
        
        return intersect_paths(obstacle_paths,
                               #[trans.transform_path(p) for p in mpl.collections.PatchCollection(self.patches).get_paths()]
                               [trans.transform_path(p.get_path()) for p in self.body_patches]
                               )

    def collision1(self,obstacle_patches):
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans = trans1
        
        transformed_obstacle_paths = [p.get_patch_transform().transform_path(p.get_path()) for p in obstacle_patches]
        return intersect_paths(transformed_obstacle_paths,
                               #[trans.transform_path(p) for p in mpl.collections.PatchCollection(self.patches).get_paths()]
                               [trans.transform_path(p.get_path()) for p in self.body_patches]
                               )
    def collision2(self,obstacle_paths):
        #use a single ship patch
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        
        trans = trans1
        
        return intersect_paths(obstacle_paths,
                               [trans.transform_path(self.ship_patch.get_path())]
                               )
                               
    def collision3(self,shapely_multipoly):
        #broken
        trans1 = matplotlib.transforms.Affine2D().translate(self.x,self.y)
        trans = trans1
        
        xy = trans.transform_path(self.ship_exterior_path).vertices
        return shapely_multipoly.intersects(geo.Polygon(xy))
        
    def set_alpha(self,alpha):
        for p in self.patches:
            p.set_alpha(alpha)
    
    @staticmethod            
    def make_trail_plot(ax,thrust,traj):
        assert thrust.shape[0] == traj.shape[0]
        for i in xrange(thrust.shape[0]):
            a = Ship_Sprite()
            a.update_thrust(horz_thrust=thrust[i,0],
                            vert_thrust=thrust[i,1])
            a.update_pose(traj[i,0],traj[i,1])
            a.update_transform_axes(ax)
        
            a.set_alpha(0.3)
            collide = False and a.collision2(obstacle_pc.get_paths())
            if collide:
                print i,'collide'
            for p in a.patches:
                #p.set_alpha(0.6)
                if(collide):
                    p.set_color('r')
                ax.add_artist(p)                  


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
    import matplotlib.animation as animation
    import shelve

    if True:
        #    import ipdb
            #ship_shelve = shelve.open('ship.shelve')
            ship_shelve = shelve.open('rrt_2d_di_best_of_all.shelve')

        #    field_shelve = shelve.open('field1.shelve')   
        #    obstacle_paths = field_shelve['obstacle_paths']

            import ship_field
            
            print 'loading local vars',ship_shelve.keys()
            
            #to pacify the linter
            traj = None
            utraj = None
            
            for k in ship_shelve.keys():
                locals()[k]=ship_shelve[k]
        
    trail_figure = plt.figure(None)
    trail_plot = trail_figure.gca() #.subplot(111,aspect='equal')
    trail_plot.set_axis_bgcolor((.9,.9,.9)) #gray background

    #process utraj so that impulses don't totally wash out the thrust visualization.
    m = 2*np.mean(np.abs(utraj))
    utraj = np.clip(utraj,-m,m)

    #only draw these time indices    
    trail_indices = np.arange(0,traj.shape[0],2)
    max_horz_thrust = np.max(np.abs(utraj[:,0]))
    max_vert_thrust = np.max(np.abs(utraj[:,1]))
    
    trail_utraj = utraj[trail_indices]
    trail_utraj[:,0] = trail_utraj[:,0]/max_horz_thrust
    trail_utraj[:,1] = trail_utraj[:,1]/max_vert_thrust
    
    trail_traj = traj[trail_indices,2:4] #use position coordinates
    
    Ship_Sprite.make_trail_plot(trail_plot,trail_utraj,trail_traj)
        
    trail_plot.set_xlim(np.min(traj[:,2])-10,np.max(traj[:,2])+10)
    trail_plot.set_ylim(np.min(traj[:,3]-10),np.max(traj[:,3])+10)
    trail_plot.set_aspect('equal')

    trail_plot.set_xlim(-10,110)
    trail_plot.set_ylim(-10,110)

    pc = ship_field.get_patch_collection()
    pc.set_color('gray')      
    trail_plot.add_collection(pc)

    if False: #plot explored states
        s = shelve.open('kin_rrt.shelve')
        tree = s['tree']
        states = [node['state'] for node in tree.node.values()]
        states = np.array(states).T
        trail_plot.plot(states[0], states[1], 'r.', zorder=0,alpha=.4)
        s.close()

    #assert False    

    ani_fig = plt.figure(None)
    ani_ax = ani_fig.gca()
    
    ani_ax.set_xlim(np.min(traj[:,2])-10,np.max(traj[:,2])+10)
    ani_ax.set_ylim(np.min(traj[:,3]-10),np.max(traj[:,3])+10)
    ani_ax.set_aspect('equal')

    ani_ax.set_xlim(-10,110)
    ani_ax.set_ylim(-10,110)

    pc = ship_field.get_patch_collection()
    pc.set_color('gray')
    ani_ax.add_collection(pc)

    ship_sprite = Ship_Sprite()
    
    for p in ship_sprite.patches:
        ani_ax.add_artist(p)
    
    #import copy
    def update_frame(i):
        j = trail_indices[i]
        print i,j
        ship_sprite.update_pose(traj[j,2],traj[j,3])
        ship_sprite.update_thrust(utraj[j,0]/max_horz_thrust,
                                  utraj[j,1]/max_vert_thrust)
        ship_sprite.update_transform_axes(ani_ax)
        ani_ax.set_title('time index: %d'%(j))
        #ani_ax.add_artist(copy.copy(ship_sprite.ship_collection))
     
    ani = animation.FuncAnimation(fig=ani_fig,func=update_frame,frames=trail_indices.size,interval=50)
    #ani.save('test.mp4', fps=20, codec='mpeg4', clear_temp=True)
