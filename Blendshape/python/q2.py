import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

def genMesh(MU, U, W):
    V = MU + U @ W
    V = np.reshape(V, (int(len(V)/3), 3))
    return V

def dispVert(shp1, shp2, color1, color2):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(shp1[:, 0], shp1[:, 1], shp1[:, 2], color=color1, s=1.0)
    ax.scatter3D(shp2[:, 0], shp2[:, 1], shp2[:, 2], color=color2, s=1.0)
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
    e0[0] = 0.0
    e0[1] = 0.0

    # Create rotation matrix from Euler angles
    R = Rotation.from_euler('xyz', [np.pi/6, np.pi/6, 0]).as_matrix()

    t = np.array([0.5, 0, 0])

    # Generate the mesh
    V0 = genMesh(B0, B, e0)

    # Apply the rotation and translation
    V1 = V0 @ R + t 
    
    # Plot
    dispVert(V0, V1, [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5])