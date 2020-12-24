addpath('functions_test\');

opts.epsilon = 1e-6;
opts.H_epsilon = 1e-14;
opts.rou = 1;
opts.maxit = 400;

opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;

f_now.grad = @df;f_now.obj = @f;
x0 = [-10,0]';
x_end = BFGS(f_now,x0,opts)
