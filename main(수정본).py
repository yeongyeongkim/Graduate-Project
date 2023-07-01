#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from utils import draw3d, read_obj, draw2d, get_projection

def Projection_3Dto2D(points_3d, P):
    points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    points_2d_homo = np.dot(points_3d_homo, P.T)
    vertices2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:]
    return vertices2d
   

if __name__ == "__main__":
    # Parameters
    ax = 1                      # scale factor
    ay = 1                      # scale factor
    x0 = 0                      # principal point
    y0 = 0                      # principal point
    skew = 0                    # skew parameter
    roll = 0                   # rotation
    pitch = 0                   # rotation
    yaw = 0                     # rotation
    translation = [0, 0, 500]     # translation : camera location
    
    projection = get_projection(ax, ay, x0, y0, skew, roll, pitch, yaw, translation)
    
    vertices3d, faces = read_obj("cat.obj")

    draw3d(vertices3d, faces)

    vertices2d = Projection_3Dto2D(vertices3d, projection)

    draw2d(vertices2d, faces)

