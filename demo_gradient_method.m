% Add folder to path
addpath(genpath('method'));

% Remove folder from path
% rmpath('lib');

%%%%%%%%%%%%%%%%%%%
% Declare Options %
%%%%%%%%%%%%%%%%%%%

% GM
opts.gm.maxit = 20000;
opts.gm.tol = 1e-8;
opts.gm.display = true;
opts.gm.step_size_method = "exact";
opts.gm.plot = false;
opts.gm.print = true;

% Exact Line Search
opts.exact.maxit = 100;
opts.exact.tol = 1e-6;
opts.exact.display = false;
opts.exact.xl = 0;
opts.exact.xr = 2;

% Armijo
opts.armijo.maxit = 100;
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;
% Diminishing
opts.diminishing.p = 2;


%%%%%%%%%%%%
% Define f %
%%%%%%%%%%%%

f.obj = @(x) (3 + x(1) + ((1-x(2))*x(2)-2)*x(2))^2 + (3 + x(1) + (x(2)-3)*x(2))^2;
f.grad = @(x) [4*x(1) + 2*x(2)*(x(2) - 3) - 2*x(2)*(x(2)*(x(2) - 1) + 2) + 12;2*(2*x(2) - 3)*(x(1) + x(2)*(x(2) - 3) + 3) - 2*(x(2)*(2*x(2) - 1) + x(2)*(x(2) - 1) + 2)*(x(1) - x(2)*(x(2)*(x(2) - 1) + 2) + 3)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Method %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.gm.step_size_method = "armijo";
opts.gm.print = true;
opts.gm.plot = false;
[x, ks, ngs, norms] = gradient_method(f,[0;0],opts);


%%%%%%%%%%%%%%%%%%%%
% Convergence plot %
%%%%%%%%%%%%%%%%%%%%
opts.gm.print = false;
opts.gm.plot = false;
figure("Name", "Convergence plot");

plot(ks,log(ngs));
hold on

plot(ks,log(norms));
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contour Plot and Trace Plot %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate random number between 10 and 20
rng(0,'twister');
x1s = [(10-(-10))*rand(4,1) + (-10); (10-(-10))*rand(4,1) + (-10); [-10 -10 -10 -10 10 10 10 10]'];
x2s = [[-2 -2 -2 -2 2 2 2 2]'; (2-(-2))*rand(4,1) + (-2); (2-(-2))*rand(4,1) + (-2)];
ps = [x1s x2s];

% Contour Plot
x1_range = linspace(-10,10);
x2_range = linspace(-2,2);
[X,Y] = meshgrid(x1_range,x2_range);
Z = (3 + X + ((1-Y).*Y-2).*Y).^2 + (3 + X + (Y-3).*Y).^2;
figure("Name", "Trace Plot");
contour(X,Y,Z,20)
hold on


% Trace Plot
opts.gm.plot = true;
opts.gm.print = false;
for j = 1:size(ps,1)
    plot(ps(j, 1),ps(j, 2),'.','MarkerSize',10);
    [x, ks, ngs, norms] = gradient_method(f,ps(j, :)',opts);
    plot(x(1),x(2),'.','MarkerSize',10);
end

hold off