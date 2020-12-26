function [x,ks,ngs] = agm_unknown(f,x0,opts)

% import
addpath(genpath('search'));

x = x0;
prev_x = x0;
t = 1;
prev_t = t;
k = 0;
prev_alpha = 0.1;
eta = 0.5;

for k = 1:opts.agm.maxit
    
    beta = t^(-1) * (prev_t - 1);
    y = x + beta * (x - prev_x);
    prev_x = x;
    alpha = prev_alpha;
    grad = f.grad(y,opts);
    x = y - alpha * grad;

    while(f.obj(x,opts)-f.obj(y,opts) > -alpha/2*norm(grad)^2)
        alpha = eta * alpha;
        x = y - alpha * grad;
    end
    t = (1/2)*(1 + sqrt(1+4*prev_t^2));
    prev_alpha = alpha;

    obj_val = f.obj(x,opts);
    
    % Calculate new norm
    ng = norm(grad);
    
    ks(k) = k;
    ngs(k) = ng;
    
    if opts.agm.print
        fprintf('k=[%5i] ; obj_val=%1.6f ; ng=%1.4e ; alpha=%1.2f ; beta=%1.2f\n',k,obj_val,ng,alpha,beta);
        x
    end
    
    % Check if stopping criteria is satisfied
    if ng <= opts.agm.tol
        break
    end
    
end