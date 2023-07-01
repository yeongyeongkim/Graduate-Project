close all; clear;
load('Blendshape_5.mat');
min_options = optimoptions(@lsqnonlin, 'Display', 'off');

IDX_3D_LANDMARKS = [4396,4370,9224,2015,3185,6073,8832,8797];
IDX_2D_LANDMARKS = [37 40 43 46 49 52 55 58];

B0 = M{1};
n_exps = length(M) - 1;
n_verts = size(B0, 1);

B0 = double(reshape(B0, [n_verts*3, 1]));
B = zeros(n_verts*3, n_exps);

for i = 1:n_exps
    B(:, i) = double(reshape(M{i+1}, [n_verts*3, 1])) - B0;
end

e0 = zeros(n_exps, 1);
I = imread('3243602421_1.jpg');
LM = readPts('3243602421_1.pts');
V0 = genMesh(B0, B, e0);


%%%%%%%%%%%%%%%%%%%%%% TEST DECODE PARAM AND PROJECTION%%%%%%%%%%%%%%%%%%%%%
solParam = [42.4292622207478;1.38295213352901;5.70977647829167;1.34452149982722;23.8782238019650;472.766577762067;615.716576482852;0];
[f, R, t] = decodeParam(solParam);
V = f * V0 * R + t;
figure('Name', 'Test1-1');
imshow(I);
hold on;
scatter(LM(IDX_2D_LANDMARKS, 1), LM(IDX_2D_LANDMARKS, 2), 'r');
scatter(V(IDX_3D_LANDMARKS, 1), V(IDX_3D_LANDMARKS, 2), 'g');

figure('Name', 'Test1-2');
imshow(I);
hold on;
scatter(LM(:, 1), LM(:, 2), 'r');
scatter(V(:, 1), V(:, 2), 'g', '.');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%% TEST ENDOCE PARAM AND PROJECTION%%%%%%%%%%%%%%%%%%%%%
solParam = encodeParam(f, R, t);
[f, R, t] = decodeParam(solParam);
V = f * V0 * R + t; %% V = Weakly Projection of V0
figure('Name', 'Test2-1');
imshow(I);
hold on;
scatter(LM(IDX_2D_LANDMARKS, 1), LM(IDX_2D_LANDMARKS, 2), 'r');
scatter(V(IDX_3D_LANDMARKS, 1), V(IDX_3D_LANDMARKS, 2), 'g');

figure('Name', 'Test2-2');
imshow(I);
hold on;
scatter(LM(:, 1), LM(:, 2), 'r');
scatter(V(:, 1), V(:, 2), 'g', '.');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = 1.0;
R = eye(3);
t = [0, 0, 0];

initialParam = encodeParam(f, R, t);
V = f * V0 * R + t; %% V = Weakly Projection of V0

figure('Name', 'Initial1');
imshow(I);
hold on;
scatter(LM(IDX_2D_LANDMARKS, 1), LM(IDX_2D_LANDMARKS, 2), 'r');
scatter(V(IDX_3D_LANDMARKS, 1), V(IDX_3D_LANDMARKS, 2), 'g');

figure('Name', 'Initial2');
imshow(I);
hold on;
scatter(LM(:, 1), LM(:, 2), 'r');
scatter(V(:, 1), V(:, 2), 'g', '.');


optimizedParam = lsqnonlin(@(x) CostFunc(x, V0(IDX_3D_LANDMARKS, :), LM(IDX_2D_LANDMARKS, :)), initialParam, [],[], min_options);

[f, R, t] = decodeParam(optimizedParam);
V = f * V0 * R + t; 

figure('Name', 'Optimized1');
imshow(I);
hold on;
scatter(LM(IDX_2D_LANDMARKS, 1), LM(IDX_2D_LANDMARKS, 2), 'r');
scatter(V(IDX_3D_LANDMARKS, 1), V(IDX_3D_LANDMARKS, 2), 'g');

figure('Name', 'OptimizedDispface');
imshow(I);
hold on;
Vd = V; Vd(:, 3) = -Vd(:, 3);
dispFace(Vd, F, [0.8, 0.8, 0.8]);

figure('Name', 'Optimized2');
imshow(I);
hold on;
scatter(LM(:, 1), LM(:, 2), 'r');
scatter(V(:, 1), V(:, 2), 'g', '.');


function E = CostFunc(P, V0, Ltar)
    [f, R, t] = decodeParam(P);
    V = f * V0 * R + t;
    E = Ltar - V(:, 1:2);
end

function P = encodeParam(f, R, t)
    P = [f; rotm2quat(R)'; t(:)];
end

function [f, R, t] = decodeParam(P)
    f = P(1);
    R = quat2rotm(P(2:5)');
    t = P(6:8)';
end

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