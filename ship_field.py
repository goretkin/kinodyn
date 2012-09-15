from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

from matplotlib.collections import PatchCollection

import math

def ellipse(center,radii):
    assert len(center)==2
    assert len(radii)==2
    sides = 40
    rad = 2*math.pi/sides
    return [ ( center[0]+radii[0]*math.cos(n*rad),center[1]+radii[1]*math.sin(n*rad)) for n in range(sides) ]

def circle(center,radius):
    return ellipse(center,(radius,radius))

EASY = False
   
obstacles_coords = []


#bounding around
obstacles_coords.append( [(-10,-10),(-10,110),(-5,110),(-5,-5),(110,-5),(110,-10)] )
obstacles_coords.append( [(110,110),(110,-10),(105,-10),(105,105),(-10,105),(-10,110)] )

if not EASY:
    #narrow passage
    w= -10
    v= +5
    x= +5
    #obstacles_coords.append( [(0,100),(10-w,100),(70-w,40),(60,40)] )
    #obstacles_coords.append( [(90,0),(80+w,0),(30+w,50),(40,50)] )

    obstacles_coords.append( [(0,100+x),(10,100+x),(70,40+x),(60,40+x)] )
    obstacles_coords.append( [(90+w,0-v),(80+w,0-v),(30+w,50-v),(40+w,50-v)] )

    #start and goal TRAPezoids
    #obstacles_coords.append(  [(30,10),(40,10),(10,40),(10,30),(30,10)] )
    #obstacles_coords.append(  [(90,70),(90,80),(80,90),(70,90),(90,70)] )

if EASY:
    obstacles_coords.append( circle((50,50),10) )
    obstacles_coords.append( circle((20,80),20) )
    obstacles_coords.append( ellipse((70,15),(15,8)) )
    obstacles_coords.append( circle((70,80),5) )
    obstacles_coords.append( circle((85,50),8) )


obstacles_polys = [Polygon(obs) for obs in obstacles_coords]

obstacles_multipoly = Polygon()

for poly in obstacles_polys:
    obstacles_multipoly = obstacles_multipoly.union(poly)


def get_patch_collection():
    obstacles_patches = [PolygonPatch(poly) for poly in obstacles_polys]
    obstacle_patch_collection = PatchCollection(obstacles_patches)    
    return obstacle_patch_collection



