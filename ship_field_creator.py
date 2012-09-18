# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:29:30 2012

@author: gustavo

based off: http://matplotlib.sourceforge.net/examples/event_handling/path_editor.py
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

class PathCreator:
    def __init__(self,ax,finished_path_callback):
        """
        finished_path_callback is called when a closed path is generated
        """
        self.finished_path_callback = finished_path_callback
        self.ax = ax
        self.canvas = self.ax.figure.canvas

        #self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        
        #outline of path
        self.line, = ax.plot([],[],marker='o', markerfacecolor='r', animated=True)
        
        self.reset()

    def reset(self):
        self.line.set_visible(False)
        self.pathdata = []
        self.pathxy = []
        self.pathpatch = None
        self.startxy = None             #starting point in data coords
        self.startxy_display = None     #starting point in display coordinates
        self.done = False               #done creating a path
        
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        if(self.pathpatch is not None):
            self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
    def button_press_callback(self,event):
        """
        returns True if button_press was handled
        """
        if event.inaxes==None: return False
        if event.button == 3:
            print "CLEAR PATH"
            self.reset()
            self.draw_callback(None)
            return True
        if event.button != 1: return False
        if self.done : return False
        
        #xydata = self.ax.transData.inverted().transform((event.x,event.y))
        
        close = False #poly is not yet closed
        if len(self.pathdata) == 0:
            #chose first vertex of polygon
            self.startxy = (event.xdata,event.ydata)
            self.startxy_display = (event.x,event.y)
            self.pathdata.append((Path.MOVETO,self.startxy))
        else:
            d = ((event.x - self.startxy_display[0])**2 + (event.y-self.startxy_display[1])**2) **(0.5)
            if(d<5):    #clicked within 5 pixels of starting point
                self.pathdata.append((Path.CLOSEPOLY,self.startxy))
                close = True
            else:
                self.pathdata.append((Path.LINETO,(event.xdata,event.ydata)))
        
        if not close:
            self.pathxy.append([event.xdata,event.ydata])
        else:
            self.pathxy.append(list(self.startxy))
        
        self.line.set_data([x for (x,y) in self.pathxy],
                            [y for (x,y) in self.pathxy])

        if close:
            self.line.set_visible(False)
        else:
            self.line.set_visible(True)
            
        codes, verts = zip(*self.pathdata)
        
        path = mpath.Path(verts, codes)

        if(self.pathpatch is not None):
            self.pathpatch.remove() 

        self.pathpatch = mpatches.PathPatch(path, facecolor='green', edgecolor='yellow', alpha=0.5)            
            
        self.ax.add_patch(self.pathpatch)
                            
        #redraw
        self.canvas.restore_region(self.background)
        if(self.pathpatch is not None):
            self.ax.draw_artist(self.pathpatch)
            
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
        if close:
            self.finished_path_callback(self.pathpatch)
            self.reset() #prepare to draw a new polygon
        
        return True
                                   
class PathInteractor:
    """
    A path editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    """

    
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, pathpatch):
        self.showverts = True

        self.ax = pathpatch.axes
        canvas = self.ax.figure.canvas
        self.canvas = canvas
        self.pathpatch = pathpatch
        self.pathpatch.set_animated(False)

        x, y = zip(*self.pathpatch.get_path().vertices)

        self.line, = ax.plot(x,y,color='y',marker='o', markerfacecolor='r', animated=True)
        #pathpatch.axes.add_line(self.line)
        self.canvas.draw()
        
        self._ind = None # the active vert

        self.canvas.mpl_connect('draw_event', self.draw_callback)
        
        #canvas.mpl_connect('button_press_event', self.button_press_callback)
        #canvas.mpl_connect('key_press_event', self.key_press_callback)
        #canvas.mpl_connect('button_release_event', self.button_release_callback)
        #canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        
        
        #for mouse drag
        self.last_displacement = None
        self.clickxy = None
        
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        print id(self),'snapshot'
        for p in ax.patches:
            self.ax.draw_artist(p)
        #self.ax.draw_artist(self.pathpatch)
        for l in ax.lines:
            self.ax.draw_artist(l)
        #self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        #pass
        
    def pathpatch_changed(self, pathpatch):
        print 'pathpatch_changed'
        'this method is called whenever the pathpatchgon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, pathpatch)
        self.line.set_visible(vis)  # don't use the pathpatch visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        # display coords
        xy = np.asarray(self.pathpatch.get_path().vertices)
        xyt = self.pathpatch.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        ind = d.argmin()

        if d[ind]>=self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes==None: return
        if event.button != 1: return
        ind = self.get_ind_under_point(event)

        self.clickxy = (event.xdata,event.ydata)
        self.last_displacement = np.array([0.0,0.0])
        
        if (ind==None):
            #check to see if clicked on poly
            #drag = self.pathpatch.contains_point(point=(event.xdata,event.ydata))
            from matplotlib._path import point_in_path
            drag = point_in_path(event.xdata,
                                 event.ydata,
                                 0,
                                 self.pathpatch.get_path(),
                                 None)                
            if drag:
                ind = 'drag_patch'
        self._ind = ind
        print self._ind

        if (self._ind is not None):
            self.pathpatch.set_animated(True)
        return (self._ind is not None) #handled event or no?

    def button_release_callback(self, event):
        #whenever a mouse button is released
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None
        self.pathpatch.set_animated(False)
        
    def key_press_callback(self, event):
        #whenever a key is pressed
        if not event.inaxes: return
        handled = False
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts: self._ind = None
            handled = True
        self.canvas.draw()
        
        return handled #handled event or no?

    def motion_notify_callback(self, event):
        #on mouse movement
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata
        
        vertices = self.pathpatch.get_path().vertices
        
        if (self._ind == 'drag_patch'):
            displacement = np.array([event.xdata,event.ydata]) - np.array(self.clickxy)
            vertices += displacement-self.last_displacement
            self.last_displacement = displacement
        else:
            vertices[self._ind] = x,y
                    
        self.line.set_data(vertices[:,0],vertices[:,1])

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

event_handling_objects = [] #objects at the beginning of list get priority
last_click = None   #element of event_handling_objects that was last sent a button_press event

def button_press_event_dispatcher(event):
    if fig.canvas.widgetlock.locked(): #matplotlib widget in use
        return
    global last_click
    last_click = None
    for o in event_handling_objects:
        if(o.button_press_callback(event)):
            #o handled the event
            last_click = o
            break
    print 'last_click',id(last_click)
        
def key_press_event_dispatcher(event):
    if(last_click):
        try:
            last_click.key_press_callback(event)
        except AttributeError:
            pass
        
def motion_notify_event_dispatcher(event):
    if(last_click):
        try:
            last_click.motion_notify_callback(event)
        except AttributeError:
            pass
        
def button_release_event_dispatcher(event):
    if(last_click):
        try:
            last_click.button_release_callback(event)
        except AttributeError:
            pass

def new_patch(patch):
    print 'new patch'
    global event_handling_objects
    interactor = PathInteractor(patch)
    event_handling_objects.insert(0,interactor) #add to beginning 

Path = mpath.Path

fig = plt.figure()

fig.canvas.mpl_connect('button_press_event', button_press_event_dispatcher)
fig.canvas.mpl_connect('key_press_event', key_press_event_dispatcher)
fig.canvas.mpl_connect('button_release_event', button_release_event_dispatcher)
fig.canvas.mpl_connect('motion_notify_event', motion_notify_event_dispatcher)
        
ax = fig.add_subplot(111)

ax.set_title('click to draw polygon')
ax.set_xlim(-10,110)
ax.set_ylim(-10,110)
ax.set_aspect('equal')
pc = PathCreator(ax,new_patch)

event_handling_objects.append(pc)

if True:
    from ship_visualize_animation import Ship_Sprite
    ship_sprite = Ship_Sprite()
    ship_sprite.update_pose(0,0,0)
    ship_sprite.update_thrust(0,0)
    ship_sprite.update_transform_axes(ax)
    for p in ship_sprite.patches:
        ax.add_artist(p)
plt.show()

import shelve
field_shelve = shelve.open('field_simple.shelve')
field_shelve['obstacle_paths']=[p.get_path() for p in ax.patches]
field_shelve.close()

