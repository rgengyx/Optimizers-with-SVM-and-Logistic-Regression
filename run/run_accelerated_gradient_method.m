% Add folder to path
addpath(genpath('../method'));
addpath(genpath('../function'));


% Global Seed Settings
rng("default");

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%

load("../small/small_dataset_sample.mat");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

% AGM
opts.agm.maxit = 20000;
opts.agm.tol = 1e-8;
opts.gm.display = true;
opts.gm.plot = false;
opts.agm.print = true;
opts.agm.beta = @beta;
opts.agm.L = 0.1;
opts.agm.step_size = step_size(opts);


%%%%%%%%%%%%%%%%%%%%%%
% Parameters Options %
%%%%%%%%%%%%%%%%%%%%%%

opts.svm.lambda = 0.1;
opts.svm.delta = 1e-1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = svm();

x0 = [10;40;10];
[x,ks,ngs] = accelerated_gradient_method(f.svm,x0,opts);


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
