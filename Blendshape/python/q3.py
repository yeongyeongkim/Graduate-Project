import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

def genMesh(MU, U, W):
    V = MU + U @ W
    V = np.reshape(V, (int(len(V)/3), 3))
    return V

def readPts(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    P = np.array([list(map(float, line.strip().split())) for line in lines])
    return P

def dispFace(shp, tl, color):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(shp[:, 0], shp[:, 1], shp[:, 2], triangles=tl, color=color)
    
    # Plot the 3D landmarks
    ax.scatter3D(V0[IDX_3D_LANDMARKS, 0], V0[IDX_3D_LANDMARKS, 1], V0[IDX_3D_LANDMARKS, 2], c='r')
    plt.show()

def dispVert(shp, color):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(shp[:, 0], shp[:, 1], shp[:, 2], color=color, s=0.5)

    # Plot the 3D landmarks
    ax.scatter3D(V0[IDX_3D_LANDMARKS, 0], V0[IDX_3D_LANDMARKS, 1], V0[IDX_3D_LANDMARKS, 2], c='r', s=50.0)
    plt.show()


if __name__ == "__main__":
    # Load the .mat file
    M = loadmat('Blendshape.mat')['M']
    F = (loadmat('Blendshape_5.mat')['F'] - 1).astype(np.int32)

    # 3D 2D landmarks
    IDX_3D_LANDMARKS = np.array([4396,4370,9224,2015,3185,6073,8832,8797]) - 1
    IDX_2D_LANDMARKS = np.array([37, 40, 43, 46, 49, 52, 55, 58]) - 1

    B0 = M[0].item()
    n_exps = len(M) - 1
    n_verts = len(B0)

    B0 = np.reshape(B0, (n_verts*3, 1))
    B = np.zeros((n_verts*3, n_exps))

    for i in range(n_exps):
        B[:, i] = (np.reshape(M[i+1].item(), (n_verts*3, 1)) - B0).squeeze()

    e0 = np.zeros((n_exps, 1))

    # TEST SOMETHING #
    e0[0] = 0.0
    e0[1] = 0.0

    # Create rotation matrix from Euler angles
    R = np.eye(3)
    t = np.array([0, 0, 0])

    # Generate the mesh
    V0 = genMesh(B0, B, e0)

    # Apply the rotation and translation
    V1 = V0 @ R + t 
    
    # Plot
    dispFace(V1, F, [0.8, 0.8, 0.8, 0.5])
    dispVert(V1, [0.0, 1.0, 0.0, 0.5])

    # Show the image and the 2D landmarks
    I = cv2.cvtColor(cv2.imread('image_006.jpg'), cv2.COLOR_BGR2RGB)
    LM = readPts('image_006.txt')

    plt.figure()
    plt.imshow(I)
    plt.scatter(LM[:, 0], LM[:, 1], c='r')
    plt.show()

    plt.figure()
    plt.imshow(I)
    plt.scatter(LM[IDX_2D_LANDMARKS, 0], LM[IDX_2D_LANDMARKS, 1], c='r')
    plt.show()