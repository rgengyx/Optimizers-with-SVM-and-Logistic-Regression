addpath('functions_test\');

opts.epsilon = 1e-6;
opts.H_epsilon = 1e-14;
opts.rou = 1;
opts.maxit = 1000;
opts.limit.step = 10;%range from [5,25]

opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;

f_now.grad = @df;f_now.obj = @f;%test function in Assignment4 2
x0 = [-10,2]';

%buffer for inspectation, no essential use
%global s_buffer;global y_buffer;global alpha_buffer; 
%s_buffer = {};y_buffer = {};alpha_buffer = {};

%see the difference between BFGS and L_BFGS
[x_end,count1] = L_BFGS(f_now,x0,opts);
[x_end2,count2] = BFGS(f_now,x0,opts);