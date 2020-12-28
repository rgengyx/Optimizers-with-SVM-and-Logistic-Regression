function [x,k,ngs] = run_accelerated_gradient_method(func, opts)

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
elseif func == "svm_sparse"
    f = svm_sparse();
elseif func == "logr"
    f = logr();
elseif func == "logr_sparse"
    f = logr_sparse();
elseif func == "logr_sgd"
    f = logr_sgd();
end

x0 = opts.x0;
[x,k,ngs] = agm_unknown(f,x0,opts);
%[x,k,ngs] = agm_known(f,x0,opts);

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