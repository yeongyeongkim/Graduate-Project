import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def genMesh(MU, U, W):
    V = MU + U @ W
    V = np.reshape(V, (int(len(V)/3), 3))
    return V

def cost_func(params, V0, Vtar):
    R = Rotation.from_euler('xyz', params[:3]).as_matrix()
    t = params[3:]
    V = V0 @ R + t
    return (Vtar - V).ravel()

def dispFace(shp1, shp2, tl, color1, color2):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(shp1[:, 0], shp1[:, 1], shp1[:, 2], triangles=tl, color=color1)
    ax.plot_trisurf(shp2[:, 0], shp2[:, 1], shp2[:, 2], triangles=tl, color=color2)
    plt.show()

def dispVert(shp1, shp2, color1, color2):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(shp1[:, 0], shp1[:, 1], shp1[:, 2], color=color1, s=0.5)
    ax.scatter3D(shp2[:, 0], shp2[:, 1], shp2[:, 2], color=color2, s=5.0)
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

    # Generate the mesh
    V0 = genMesh(B0, B, e0)

    ### Assume that we don't know the transformation of the shape V1 ###
    R = np.array([np.pi/6, -np.pi/6, np.pi/2])
    t = np.array([0.7, -0.2, 0.6])
    Vtar = V0 @ Rotation.from_euler('xyz', R).as_matrix() + t
    ####################################################################

    # We only have V1 and find R and t using V1 and V0 only
    R0 = Rotation.from_matrix(np.eye(3)).as_euler('xyz')  # Initial rotation
    t0 = np.array([0.0, 0.0, 0.0])  # Initial translation

    print('Before Optimization')
    print('Rotation Error', np.linalg.norm(R0 - R))
    print('Translation Error', np.linalg.norm(t0 - t))

    initial_param = np.concatenate([R0, t0])
    optimized_param = least_squares(cost_func, initial_param, args=(V0, Vtar)).x

    R0 = optimized_param[:3]
    t0 = optimized_param[3:]

    V1 = V0 @ Rotation.from_euler('xyz', R0).as_matrix() + t0

    print('After Optimization')
    print('Rotation Error', np.linalg.norm(R0 - R))
    print('Translation Error', np.linalg.norm(t0 - t))


    # Before optimize
    dispFace(V0, V1, F, [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5])
    dispVert(V0, V1, [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5])

    # After optimize
    dispFace(Vtar, V1, F, [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5])
    dispVert(Vtar, V1, [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5])