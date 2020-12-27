function [x,k,ngs] = run_bfgs(func, opts)

% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
% Global Seed Settings
rng("default");


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
[x,k,ngs] = BFGS(f,x0,opts);

end