% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));

% Global Seed Settings
rng("default");

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;
load("small/small_dataset_mod.mat");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

% AGM
opts.agm.maxit = 2000;
opts.agm.tol = 1e-4;
opts.gm.display = true;
opts.gm.plot = false;
opts.agm.print = true;


%%%%%%%%%%%%%%%%%%%%%%
% Parameters Options %
%%%%%%%%%%%%%%%%%%%%%%

% SVM
opts.svm.lambda = 0.1;
opts.svm.delta = 1e-1;

% Logistic Regression
opts.logr.lambda = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%f = svm();
f = logistic_regression();

x0 = [0;0;0];
[x,ks,ngs] = accelerated_gradient_method(f,x0,opts);


%%%%%%%%%%%%%%%%%
% Visualization %
%%%%%%%%%%%%%%%%%

plot_scatter(data1,label1);
hold on

x1 = -3:0.01:3;
x2 = calculate_x2(x,x1);
plot(x1,x2);


%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

function x2 = calculate_x2(x, x1)
    x2 = (-x1 * x(1) - x(3))/x(2); 
end
