close all; clear;
load('Blendshape.mat');

B0 = M{1};
n_exps = length(M) - 1;
n_verts = size(B0, 1);

B0 = reshape(B0, [n_verts*3, 1]);
B = zeros(n_verts*3, n_exps);

for i = 1:n_exps
    B(:, i) = reshape(M{i+1}, [n_verts*3, 1]) - B0;
end

e0 = zeros(n_exps, 1);

%% TEST SOMETHING %%
e0(1) = 1.0;
e0(2) = 0.5;

V0 = genMesh(B0, B, e0);
dispFace(V0, F, [0.8, 0.8, 0.8]);
figure;
scatter3(V0(:, 1), V0(:, 2), V0(:, 3), 1);


function V = genMesh(MU, U, W)
    V = MU + U * W;
    V = reshape(V, [length(V)/3 3]);
end