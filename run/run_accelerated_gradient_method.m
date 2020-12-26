% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));


% Global Seed Settings
rng("default");

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%

load("small/small_dataset_sample.mat");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

% AGM
opts.agm.maxit = 1000;
opts.agm.tol = 1e-4;
opts.gm.display = true;
opts.gm.plot = false;
opts.agm.print = true;
opts.agm.beta = @beta;
opts.agm.L = 0.1;
opts.agm.step_size = step_size(opts);


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

f = svm();
% f = logistic_regression();

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


%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

% Define Extrapolation parameter beta
function [prev_t, beta] = beta(prev_t)
    t = (1/2)*(1 + sqrt(1+4*prev_t^2));
    beta = (prev_t - 1) / t;
    prev_t = t;
end

% Step size
function step_size = step_size(opts)
    L = opts.agm.L;
    step_size = 1 / L;
end
