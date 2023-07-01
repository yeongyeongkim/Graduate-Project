import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def genMesh(MU, U, W):
    V = MU + U @ W
    V = np.reshape(V, (int(len(V)/3), 3))
    return V

def dispFace(shp, tl, color):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(shp[:, 0], shp[:, 1], shp[:, 2], triangles=tl, color=color)
    plt.show()

def dispVert(shp, color):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(shp[:, 0], shp[:, 1], shp[:, 2], color=color, s=1.0)
    plt.show()

if __name__ == "__main__":
    # Load the .mat file
    M = loadmat('Blendshape.mat')['M']
    F = (loadmat('Blendshape_5.mat')['F'] - 1).astype(np.int32)

    B0 = M[0].item()
    n_exps = len(M) - 1
    n_verts = len(B0)

    B0 = np.reshape(B0, (n_verts*3, 1))
    B = np.zeros((n_verts*3, n_exps))

    for i in range(n_exps):
        B[:, i] = (np.reshape(M[i+1].item(), (n_verts*3, 1)) - B0).squeeze()

    e0 = np.zeros((n_exps, 1))

    # TEST SOMETHING #
    e0[0] = 1.0
    e0[1] = 0.5

    V0 = genMesh(B0, B, e0)

    # Plot
    dispFace(V0, F, [0.8, 0.8, 0.8])
    dispVert(V0, [1.0, 0.0, 0.0])
