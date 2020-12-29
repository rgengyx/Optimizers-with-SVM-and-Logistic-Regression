function [x, k, ngs, train_accs, test_accs] = agm_known(f,x0,opts)

% import
addpath(genpath('search'));

x = x0;
prev_x = x0;
t0 = 1;
prev_t = t0;
k = 0;
train_accs = [];
test_accs = [];

for k = 1:opts.agm.maxit

    
    [prev_t, beta] = opts.agm.beta(prev_t);
    alpha = opts.agm.step_size;
    y = x + beta * (x - prev_x);
    prev_x = x;
    grad = f.grad(y,opts);
    x = y - alpha * grad;
    obj_val = f.obj(x,opts);
    
    % Calculate new norm
    ng = norm(grad);
    k = count;
    ks(k) = k;
    ngs(k) = ng;
    % test accuracy
    if opts.cr_save
        [CR_train,CR_test] = train_test_accuracy(x_now);
        train_accs(k) = CR_train;
        test_accs(k) = CR_test;
    end
    
    if opts.agm.print
        if opts.cr_save
            fprintf('k=[%5i] ; obj_val=%1.6f ; ng=%1.4e ; alpha=%1.2f ; train_acc=%1.4f ; test_acc=%1.4f\n',k,obj_val,ng,alpha,CR_train, CR_test);
        else
            fprintf('k=[%5i] ; obj_val=%1.6f ; ng=%1.4e ; alpha=%1.2f\n',k,obj_val,ng,alpha);
        end
    end
    
    % Check if stopping criteria is satisfied
    if ng <= opts.agm.tol
        break
    end
    
end