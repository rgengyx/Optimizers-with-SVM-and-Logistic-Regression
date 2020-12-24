function [x, ks, ngs, norms] = newton_method(f,x0,opts)

% import
addpath(genpath('lib'));

x = x0;
k = 0;
alpha = 0;
ks = [];
ngs = [];
norms = [];

for k = 1:opts.gm.maxit

    % Calculate Alpha 
    if strcmp(opts.gm.step_size_method, "exact")
        alpha = exact_line_search(f,x,opts);
    elseif strcmp(opts.gm.step_size_method, "armijo")
        alpha = armijo_line_search(f,x,opts);
    elseif strcmp(opts.gm.step_size_method, "diminishing")
        alpha = diminishing_step_size(k, opts.diminishing.p);
    end

    % Calculate d
    d = f.hessian(x) \ (-f.grad(x));
    
    % Calculate next x
    old_x = x;
    x = x + alpha * d;
    new_x = x;
    
    % Plot trace
    if opts.gm.plot
        plot([old_x(1) new_x(1)], [old_x(2), new_x(2)]);
    end
    
    % Calculate new Gradient 
    grad = f.grad(x);
    
    % Calculate new norm
    ng = norm(grad);
    
    % Add new element to arrays
    ks(k) = k;
    ngs(k) = ng;
    norms(k) = norm(x - [-1;1]);
    
    if opts.gm.print
        obj_val   = f.obj(x);
        fprintf(1,'[%5i] ; %1.6f ; %1.4e ; %1.2f\n',k,obj_val,ng,alpha);
    end
    
    % Check if stopping crkia is satisfied
    if ng <= opts.gm.tol
        break
    end
    
end