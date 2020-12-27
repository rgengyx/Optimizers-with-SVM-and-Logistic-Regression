function [x,k,ngs] = run_L_bfgs_sgd(func, opts)

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
elseif func == "logr"
    f = logistic_regression();
end

x0 = opts.x0;
[x,k,ngs] = L_BFGS(f,x0,opts);

end