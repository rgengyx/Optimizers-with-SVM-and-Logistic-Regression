% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;
load("small\small_dataset_mod.mat");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

% BFGS
opts.lbfgs.epsilon = 1e-6;
opts.lbfgs.H_epsilon = 1e-14;
opts.lbfgs.rou = 1;
opts.lbfgs.maxit = 400;
opts.lbfgs.limit_step = 10;%range [5,25]

% Armijo
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;


% Sample
opts.sample.m = length(data1);


%%%%%%%%%%%%%%%%%%%%%%
% Parameters Options %
%%%%%%%%%%%%%%%%%%%%%%

% SVM
opts.svm.lambda = 1/opts.sample.m;
opts.svm.delta = 1e-4;

% Logistic Regression
opts.logr.lambda = 0.1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% f = svm();
f = logistic_regression();
%f = bfgs();

x0 = [0;0;0];
[x,k] = L_BFGS(f,x0,opts);


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

