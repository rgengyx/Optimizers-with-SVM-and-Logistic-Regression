function [x, ks, ngs, norms] = newton_global(f,x0,opts)

% import
addpath(genpath('search'));

x = x0;
k = 0;
alpha = 0;
ks = [];
ngs = [];
norms = [];

for k = 1:opts.newton.maxit

    % Calculate d
    d = f.hessian(x) \ (-f.grad(x));
    
    % If d is a good descent direction
    if (-f.grad(x)' * d < opts.newton.beta1 * min(1, norm(d)^opts.newton.beta2) * norm(d)^2) || (f.grad(x)' * d >= 0)
        d = f.grad(x);
    end
    
    % Calculate Alpha
    if strcmp(opts.newton.step_size_method, "exact")
        alpha = exact_line_search(f,x,opts);
    elseif strcmp(opts.newton.step_size_method, "armijo")
        alpha = armijo_line_search(f,x,d,opts);
    elseif strcmp(opts.newton.step_size_method, "diminishing")
        alpha = diminishing_step_size(k, opts.diminishing.p);
    end
    
    % Calculate next x
    old_x = x;
    x = x + alpha * d;
    new_x = x;
    
    % Plot trace
    if opts.newton.plot
        plot([old_x(1) new_x(1)], [old_x(2), new_x(2)]);
    end
    
    % Calculate new Gradient 
    grad = f.grad(x);
    
    % Calculate new norm
    ng = norm(grad);
    
    % Add new element to arrays
    ks(k) = k;
    ngs(k) = ng;
    norms(k) = norm(x - [1;1]);
    
    if opts.newton.print
        obj_val   = f.obj(x);
        fprintf(1,'[%5i] ; %1.6f ; %1.4e ; %1.2f\n',k,obj_val,ng,alpha);
    end
    
    % Check if stopping crkia is satisfied
    if ng <= opts.newton.tol
        break
    end
    
end