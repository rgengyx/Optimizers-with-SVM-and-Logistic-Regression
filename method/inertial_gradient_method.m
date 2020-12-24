function [x,ks,ngs] = inertial_gradient_method(f,x0,opts)

% import
addpath(genpath('search'));

x = x0;
prev_x = x0;
t0 = 1;
prev_t = t0;
k = 0;
beta = opts.ls.beta;
l = opts.ls.l;
alpha = 1.99 * (1-beta)/ l;

for k = 1:opts.agm.maxit
    
    y = x + beta * (x - prev_x);
    prev_x = x;
    grad = f.grad(x,opts);
    x = y - alpha * grad;
    obj_val = f.obj(x,opts);
    
    % Calculate new norm
    ng = norm(grad);
    
    ks(k) = k;
    ngs(k) = ng;
    
    while f.obj(x,opts) - f.obj(prev_x,opts) > f.grad(prev_x,opts)'*(x-prev_x)+(l/2)*norm(x-prev_x)^2
        l = 2 * l;
        alpha = 1.99*(1-beta)/l;
        x = y - alpha * f.grad(prev_x,opts);
    end
    
    if opts.agm.print
        fprintf('k=[%5i] ; obj_val=%1.6f ; ng=%1.4e ; alpha=%1.2f ; beta=%1.2f\n',k,obj_val,ng,alpha,beta);
    end

    % Check if stopping criteria is satisfied
    if ng <= opts.agm.tol
        break
    end
    
end