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
f = func();

x0 = opts.x0;
[x,k,ngs] = BFGS(f,x0,opts);

end