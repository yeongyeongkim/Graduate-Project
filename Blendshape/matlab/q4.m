close all; clear;
load('Blendshape.mat');

B0 = double(M{1});
n_exps = length(M) - 1;
n_verts = size(B0, 1);

B0 = reshape(B0, [n_verts*3, 1]);
B = zeros(n_verts*3, n_exps);

for i = 1:n_exps
    B(:, i) =double(reshape(M{i+1}, [n_verts*3, 1]) - B0);
end

e0 = zeros(n_exps, 1);

%% TEST SOMETHING %%
e0(1) = 1.0;
e0(2) = 0.5;
V0 = genMesh(B0, B, e0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ASSUME THAT WE DON'T KNOW THE TRANSFORMATION OF THE SHAPE V1 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = [pi/6, -pi/6, pi/2];
t = [0.7, -0.2, 0.6];
Vtar = V0 * eul2rotm(R) + t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% WE ONLY HAVE V1 AND FIND R and t using V1 and V0 ONLY %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


min_options = optimoptions(@lsqnonlin, 'Display', 'off');
    %'Algorithm', 'levenberg-marquardt',... 
    %'Display', 'off',...);
    %'UseParallel', true);




R0 = rotm2eul(eye(3, 3)); % Initial Rotation
t0 = [0.0, 0.0, 0.0]; % Initial Translation

figure('Name', 'Initial');
scatter3(Vtar(:, 1), Vtar(:, 2), Vtar(:, 3), 1);
hold on;
scatter3(V0(:, 1), V0(:, 2), V0(:, 3), 1);

fprintf('Before Optimzation\n');
fprintf('Rotation Error %f\n', norm(R0 - R));
fprintf('Translation Error %f\n', norm(t0 - t));




initialParam = [R0(:); t0(:)];

optimizedParam = lsqnonlin(@(x) CostFunc(x, V0, Vtar), initialParam, [],[], min_options);


R0 = optimizedParam(1:3)';
t0 = optimizedParam(4:6)';


V1 = V0 * eul2rotm(R0) + t0;


figure('Name', 'After');
scatter3(Vtar(:, 1), Vtar(:, 2), Vtar(:, 3), 1);
hold on;
scatter3(V1(:, 1), V1(:, 2), V1(:, 3), 5);

fprintf('After Optimzation\n');
fprintf('Rotation Error %f\n', norm(R0 - R));
fprintf('Translation Error %f\n', norm(t0 - t));


function V = genMesh(MU, U, W)
    V = MU + U * W;
    V = reshape(V, [length(V)/3 3]);
end

function E = CostFunc(Param, V0, Vtar)
    R = Param(1:3)';
    t = Param(4:6)';
    V = V0 * eul2rotm(R) + t;
    E = Vtar - V;
end
