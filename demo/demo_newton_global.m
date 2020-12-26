% Add files to path
addpath(genpath('method'));

%%%%%%%%%%%%%%%%%%%
% Declare Options %
%%%%%%%%%%%%%%%%%%%

% Newton
opts.newton.maxit = 20000;
opts.newton.tol = 1e-8;
opts.newton.display = true;
opts.newton.step_size_method = "armijo";
opts.newton.plot = false;
opts.newton.print = true;
opts.newton.beta1 = 1e-6;
opts.newton.beta2 = 0.1;

% Armijo
opts.armijo.maxit = 100;
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;


%%%%%%%%%%%%
% Define f %
%%%%%%%%%%%%

f.obj = @(x) (3 + x(1) + ((1-x(2))*x(2)-2)*x(2))^2 + (3 + x(1) + (x(2)-3)*x(2))^2;
f.grad = @(x) [4*x(1) + 2*x(2)*(x(2) - 3) - 2*x(2)*(x(2)*(x(2) - 1) + 2) + 12;2*(2*x(2) - 3)*(x(1) + x(2)*(x(2) - 3) + 3) - 2*(x(2)*(2*x(2) - 1) + x(2)*(x(2) - 1) + 2)*(x(1) - x(2)*(x(2)*(x(2) - 1) + 2) + 3)];
f.hessian = @(x) [4,4*x(2) - 2*x(2)*(2*x(2) - 1) - 2*x(2)*(x(2) - 1) - 10;4*x(2) - 2*x(2)*(2*x(2) - 1) - 2*x(2)*(x(2) - 1) - 10,4*x(1) - 2*(6*x(2) - 2)*(x(1) - x(2)*(x(2)*(x(2) - 1) + 2) + 3) + 2*(x(2)*(2*x(2) - 1) + x(2)*(x(2) - 1) + 2)^2 + 2*(2*x(2) - 3)^2 + 4*x(2)*(x(2) - 3) + 12];


%%%%%%%%%%%%%%%%%
% Call Function %
%%%%%%%%%%%%%%%%%

opts.gm.step_size_method = "armijo";
opts.newton.print = true;
[x, ks, ngs, norms] = newton_global(f,[0,0],opts);


%%%%%%%%%%%%%%%%%%%%
% Convergence plot %
%%%%%%%%%%%%%%%%%%%%

opts.gm.print = false;
opts.newton.plot = false;
figure("Name", "Convergence plot");

plot(ks,log(ngs),'Color',"red");
hold on

plot(ks,log(norms),'Color',"blue");
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
figure("Name", "Contour Plot");
contour(X,Y,Z,20)
hold on

% Trace plot
opts.newton.plot = true;
opts.newton.print = false;
for j = 1:size(ps,1)
    plot(ps(j, 1),ps(j, 2),'.','MarkerSize',20);
    [x, ks, ngs, norms] = newton_global(f,ps(j, :)',opts);
    plot(x(1),x(2),'.','MarkerSize',20); 
end

hold off