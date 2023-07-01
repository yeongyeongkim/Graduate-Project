#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def Projection_3Dto2D(points_3d, P):
    points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    points_2d_homo = np.dot(points_3d_homo, P.T)
    vertices2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:]
    return vertices2d
   

def get_projection(ax, ay, x0, y0, skew, roll, pitch, yaw, translation):
    # Define rotation matrices
    Rx = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)

    # Define projection matrix
    K = np.array([[ax, skew, x0],
                [0, ay, y0],
                [0, 0, 1]])
    P = np.dot(K, np.hstack((R, np.array(translation).reshape((3, 1)))))
    return P

def read_obj(filename):
    with open(filename) as file:
        content = file.readlines()
        v = []
        vn = []
        vt = []
        tmp = []
        for i in range(len(content)):
            line = content[i]
            # if line[0] == '#' or 'o':
            #     continue

            if line[0:2] == 'vn':
                v_normal_info = line.replace(' \n', '').replace('vn ', '')
                vn.append(np.array(v_normal_info.split(' ')).astype(float))
            elif line[0:2] == 'v ':
                vertex_info = line.replace(' \n', '').replace('v ', '')
                v.append(np.array(vertex_info.split(' ')).astype(float))
            elif line[0:2] == 'vt':
                texture_info = line.replace(' \n', '').replace('vt ', '')
                vt.append(np.array(texture_info.split(' ')).astype(float))
            elif line[0:2] == 'f ':
                face_info = line.replace(' \n', '').replace('f ', '')
                for i in range(3):
                    tmp.append(np.array(face_info.split(' ')[i].split('/')).astype(int))
        tmp = np.array(tmp).reshape((-1, 3, 3))

    return np.array(v), np.array(tmp)

def draw2d(vertices2d, faces):
    obj_f_v = faces.reshape((-1, 3))
    connectivity = np.zeros((obj_f_v.shape[0] * obj_f_v.shape[1], 2), dtype=int)
    for i in range(0, obj_f_v.shape[0]):
        connectivity[i*3, 0] = obj_f_v[i, 0]
        connectivity[i*3, 1] = obj_f_v[i, 1]
        connectivity[i*3-1, 0] = obj_f_v[i, 1]
        connectivity[i*3-1, 1] = obj_f_v[i, 2]
        connectivity[i*3-2, 0] = obj_f_v[i, 2]
        connectivity[i*3-2, 1] = obj_f_v[i, 0]
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    xs = vertices2d[:, 0]
    ys = vertices2d[:, 1]
    ax.scatter(xs, ys, color='blue')
    for i in range(0, connectivity.shape[0]):
        a0 = vertices2d[connectivity[i, 0]-1, :]
        a1 = vertices2d[connectivity[i, 1]-1, :]
        ax.plot((a0[0], a1[0]), (a0[1], a1[1]), 'r', linewidth=1)
    plt.show()
    

def draw3d(vertices, faces):
    faces = faces.reshape(-1, 3) - 1

    # Create figure and 3D axis
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D mesh with edge lines
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='none', edgecolor='black', linewidth=0.5)

    # Set axis limits and labels
    ax.set_xlim3d(-600, 600)
    ax.set_ylim3d(-100, 500)
    ax.set_zlim3d(-200, 200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set same scale for all three axes
    ax.set_box_aspect([1, 1, 1])

    # Show plot
    plt.show()

    

