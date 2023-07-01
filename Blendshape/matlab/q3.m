close all; clear;
load('Blendshape.mat');

IDX_3D_LANDMARKS = [4396,4370,9224,2015,3185,6073,8832,8797];
IDX_2D_LANDMARKS = [37 40 43 46 49 52 55 58];

B0 = M{1};
n_exps = length(M) - 1;
n_verts = size(B0, 1);

B0 = reshape(B0, [n_verts*3, 1]);
B = zeros(n_verts*3, n_exps);

for i = 1:n_exps
    B(:, i) =reshape(M{i+1}, [n_verts*3, 1]) - B0;
end

e0 = zeros(n_exps, 1);

%% TEST SOMETHING %%
e0(1) = 1.0;
e0(2) = 0.5;


R = eye(3);
t = [0, 0, 0];

V0 = genMesh(B0, B, e0) * R + t;
dispFace(V0, F, [0.8, 0.8, 0.8]);
hold on;
scatter3(V0(IDX_3D_LANDMARKS, 1), V0(IDX_3D_LANDMARKS, 2), V0(IDX_3D_LANDMARKS, 3), 50, 'r', 'filled');

figure;
I = imread('image_006.jpg');
LM = readPts('image_006.pts');
imshow(I);
hold on;
scatter(LM(:, 1), LM(:, 2), 'r');

figure;
imshow(I);
hold on;
scatter(LM(IDX_2D_LANDMARKS, 1), LM(IDX_2D_LANDMARKS, 2), 'r');


function V = genMesh(MU, U, W)
    V = MU + U * W;
    V = reshape(V, [length(V)/3 3]);
end

function P = readPts(F)
    F = fopen(F, 'r');
    fgets(F); fgets(F); fgets(F);
    P = fscanf(F, '%f %f', [2 Inf]);
    fclose(F);
    P = P';
end