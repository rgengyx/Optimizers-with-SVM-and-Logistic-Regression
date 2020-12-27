function opts = config(opts)

%%%%%%%%%%%%%%%%%%%%%
% Method Parameters %
%%%%%%%%%%%%%%%%%%%%%

% GM
opts.gm.maxit = 3000;
opts.gm.tol = 1e-4;
opts.gm.display = true;
opts.gm.step_size_method = "armijo";
opts.gm.plot = false;
opts.gm.print = true;
opts.gm.batch_size = 100;
opts.gm.batch_epsilon = 1e-2;

% AGM
opts.agm.maxit = 3000;
opts.agm.tol = 1e-4;
opts.gm.display = true;
opts.gm.plot = false;
opts.agm.print = false;
opts.agm.L = 2;
opts.agm.batch_size = 100;
opts.agm.batch_epsilon = 1e-2;

% BFGS
opts.bfgs.epsilon = 1e-8;
opts.bfgs.H_epsilon = 1e-14;
opts.bfgs.rou = 1;
opts.bfgs.maxit = 200;
opts.bfgs.batch_size = 200;
opts.bfgs.batch_epsilon = 1e-2;

% LBFGS
opts.lbfgs.epsilon = 1e-8;
opts.lbfgs.H_epsilon = 1e-14;
opts.lbfgs.rou = 1;
opts.lbfgs.maxit = 400;
opts.lbfgs.limit_step = 10;%range [5,25]
opts.lbfgs.batch_size = 100;
opts.lbfgs.batch_epsilon = 1e-1;

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

% SVM
opts.svm.lambda = 1/opts.sample.m;
opts.svm.delta = 1e-4;

% Logistic Regression
opts.logr.lambda = 0.1;


end