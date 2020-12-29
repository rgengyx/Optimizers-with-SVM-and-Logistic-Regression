function [x, k, ngs, train_accs, test_accs] = agm_unknown(f,x0,opts)

x = x0;
prev_x = x0;
t = 1;
prev_t = t;
k = 0;
prev_alpha = 0.1;
eta = 0.5;
train_accs = [];
test_accs = [];

for k = 1:opts.agm.maxit
    beta = t^(-1) * (prev_t - 1);
    y = x + beta * (x - prev_x);
    prev_x = x;
    alpha = prev_alpha;
    %opts.sgd_ratio = (1 + 99 * opts.sgd_ratio) / 100;%here use the diminish sgd ratio
    grad = f.grad(y,opts);
    if norm(grad) < opts.agm.sgd_epsilon;
        opts.sgd_ratio = 1;
    end
    x = y - alpha * grad;

    while(f.obj(x,opts)-f.obj(y,opts) > -alpha/2*norm(grad)^2)
        alpha = eta * alpha;
        x = y - alpha * grad;
    end
    
    t_next = (1/2)*(1 + sqrt(1+4*t^2));
    prev_t = t;
    t = t_next;
    
    prev_alpha = alpha;

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