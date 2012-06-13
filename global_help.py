# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:14:05 2012

@author: gustavo

"""
import matplotlib as mpl
import types #for monkey patching

def Rectangle_centered(xy,width,height,*args,**kwargs):
    x = xy[0] - width/2
    y = xy[1] - height/2
    rect = mpl.patches.Rectangle((x,y),width,height,*args,**kwargs)
    rect._x_centered = 0
    rect._y_centered = 0
    #monkey patch some things (really should just inherit from Rectangle)
    def set_xy(self,xy):
        (self._x_centered,self._y_centered) = xy
        self._x = self._x_centered - self._width/2
        self._y = self._y_centered - self._height/2
    def set_width(self,width):
        self._width = width
        self.set_xy((self._x_centered,self._y_centered))
    def set_height(self,height):
        self._height = height
        self.set_xy((self._x_centered,self._y_centered))
        
    rect.set_xy = types.MethodType(set_xy,rect)
    rect.set_width = types.MethodType(set_width,rect)
    rect.set_height = types.MethodType(set_height,rect)
    
    return rect