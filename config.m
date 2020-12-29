function opts = config(opts)

%%%%%%%%%%%%%%%%%%%%%
% Method Parameters %
%%%%%%%%%%%%%%%%%%%%%

%if with big data, not save accuracy to speed up
opts.cr_save = false;

%sgd ratio
opts.sgd_ratio = 1 / 1.5; %set 1 if no sgd, else less than 1 for initial batch sgd

% GM
opts.gm.maxit = 500;
opts.gm.tol = 1e-4;
opts.gm.display = true;
opts.gm.step_size_method = "armijo";
opts.gm.plot = false;
opts.gm.print = true;
opts.gm.batch_size = 100;
opts.gm.batch_epsilon = 1e-2;
opts.gm.sgd_epsilon = 1e-2;

% AGM
opts.agm.maxit = 3000;
opts.agm.tol = 1e-6;
opts.gm.display = true;
opts.gm.plot = false;
opts.agm.print = true;
opts.agm.L = 2;
opts.agm.batch_size = 50;
opts.agm.batch_epsilon = 1e-2;
opts.agm.sgd_epsilon = 5 * 1e-3;

% BFGS
opts.bfgs.epsilon = 1e-6;
opts.bfgs.H_epsilon = 1e-14;
opts.bfgs.rou = 1;
opts.bfgs.maxit = 400;
opts.bfgs.batch_size = 200;
opts.bfgs.batch_epsilon = 1e-2;
opts.bfgs.print = true;

% LBFGS
opts.lbfgs.epsilon = 1e-6;
opts.lbfgs.H_epsilon = 1e-14;
opts.lbfgs.rou = 1;
opts.lbfgs.maxit = 50;
opts.lbfgs.limit_step = 5;%range [5,25]
opts.lbfgs.batch_size = 100;
opts.lbfgs.batch_epsilon = 1e-1;
opts.lbfgs.print = true;

%%%%%%%%%%%%%%%%%%%%%
% Search Parameters %
%%%%%%%%%%%%%%%%%%%%%

% Armijo
opts.armijo.maxit = 100;
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;

%%%%%%%%%%%%%%%%%%%%%%
% Parameters Options %
%%%%%%%%%%%%%%%%%%%%%%

%Sample
global data1;
sizes = size(data1);
opts.sample.m = sizes(2);
opts.split_ratio = 0.1;

% SVM
opts.svm.lambda = 1 / opts.sample.m;
opts.svm.delta = 1e-2;

% Logistic Regression
opts.logr.lambda = 0.1;


end