import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import cv2

def genMesh(MU, U, W):
    V = MU + U @ W
    V = np.reshape(V, (int(len(V)/3), 3))
    return V

def cost_func(P, V0, Ltar):
    f, R, t = decode_param(P)
    V = f * V0 @ R + t
    E = Ltar - V[:, :2]
    return E.ravel()

def encode_param(f, R, t):
    P = np.concatenate(([f], Rotation.from_matrix(R).as_quat(), t))
    return P

def decode_param(P):
    f = P[0]
    R = Rotation.from_quat(P[1:5]).as_matrix()
    t = P[5:]
    return f, R, t

def read_pts(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    P = np.array([list(map(float, line.split())) for line in lines])
    return P

def plot(LM, V, IDX_2D_LANDMARKS, IDX_3D_LANDMARKS):
    # Plot
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(I)
    ax1.scatter(LM[IDX_2D_LANDMARKS, 0], LM[IDX_2D_LANDMARKS, 1], color=[1.0, 0.0, 0.0], alpha=0.5)
    ax1.scatter(V[IDX_3D_LANDMARKS, 0], V[IDX_3D_LANDMARKS, 1], color=[0.0, 1.0, 0.0], alpha=0.5, s=0.5)

    ax2 = fig.add_subplot(122)
    ax2.imshow(I)
    ax2.scatter(LM[:, 0], LM[:, 1], color=[1.0, 0.0, 0.0], alpha=0.5)
    ax2.scatter(V[:, 0], V[:, 1], color=[0.0, 1.0, 0.0], alpha=0.5, s=0.5)
    plt.show()

if __name__ == "__main__":
    # Load the .mat file
    M = loadmat('Blendshape_5.mat')['M']
    F = (loadmat('Blendshape_5.mat')['F'] - 1).astype(np.int32)
    
    # 3D 2D landmarks
    IDX_3D_LANDMARKS = np.array([4396, 4370, 9224, 2015, 3185, 6073, 8832, 8797]) - 1
    IDX_2D_LANDMARKS = np.array([37, 40, 43, 46, 49, 52, 55, 58]) - 1
    
    B0 = M[0].item()
    n_exps = len(M) - 1
    n_verts = len(B0)

    B0 = np.reshape(B0, (n_verts*3, 1))
    B = np.zeros((n_verts*3, n_exps))

    for i in range(n_exps):
        B[:, i] = (np.reshape(M[i+1].item(), (n_verts*3, 1)) - B0).squeeze()

    e0 = np.zeros((n_exps, 1))

    I = cv2.cvtColor(cv2.imread('3243602421_1.jpg'), cv2.COLOR_BGR2RGB)
    LM = read_pts('3243602421_1.txt')

    # Generate the mesh
    V0 = genMesh(B0, B, e0)

    ### Test decode, encode params ###
    # Test decode param and projection
    sol_param = np.array([42.4292622207478,5.70977647829167,1.34452149982722,23.8782238019650,1.38295213352901,472.766577762067,615.716576482852,0])
    f, R, t = decode_param(sol_param)
    V = f * V0 @ R + t

    plot(LM, V, IDX_2D_LANDMARKS, IDX_3D_LANDMARKS)

    # Test encode param and projection
    sol_param = encode_param(f, R, t)
    f, R, t = decode_param(sol_param)
    V = f * V0 @ R + t

    plot(LM, V, IDX_2D_LANDMARKS, IDX_3D_LANDMARKS)

    ### Optimization ###
    # Create initial rotation matrix from Euler angles
    f = 1.0
    R = np.eye(3)
    t = np.array([0, 0, 0])
    initial_param = encode_param(f, R, t)
    f, R, t = decode_param(initial_param)
    V = f * V0 @ R + t
    
    # Before optimization
    plot(LM, V, IDX_2D_LANDMARKS, IDX_3D_LANDMARKS)

    # Optimize parameters
    optimized_param = least_squares(cost_func, initial_param, args=(V0[IDX_3D_LANDMARKS, :], LM[IDX_2D_LANDMARKS, :]))

    # Decode parameters
    f, R, t = decode_param(optimized_param.x)
    V = f * V0 @ R + t
    
    # After optimization
    plot(LM, V, IDX_2D_LANDMARKS, IDX_3D_LANDMARKS)



