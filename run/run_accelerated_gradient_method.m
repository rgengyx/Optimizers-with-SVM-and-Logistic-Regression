function x = run_accelerated_gradient_method(func, opts)

% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
% Global Seed Settings
rng("default");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

opts.agm.beta = @beta;
opts.agm.step_size = step_size(opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func == "svm"
    f = svm();
elseif func == "logr"
    f = logistic_regression();
end

x0 = [0;0;0];
[x,ks,ngs] = agm_unknown(f,x0,opts);
%[x,ks,ngs] = agm_known(f,x0,opts);

%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

% Define Extrapolation parameter beta
function [prev_t, beta1] = beta(prev_t)
    t = (1/2)*(1 + sqrt(1+4*prev_t^2));
    beta1 = (prev_t - 1) / t;
    prev_t = t;
end

% Step size
function step_size1 = step_size(opts)
    L = opts.agm.L;
    step_size1 = 1 / L;
end

end