function [x, k, ngs,train_accs, test_accs] = gradient_method(f,x0,opts)

x = x0;
k = 0;
alpha = 0;
ks = [];
ngs = [];
norms = [];
train_accs = [];
test_accs = [];

for k = 1:opts.gm.maxit
    
    % Calculate Gradient
    opts.sgd_ratio = (1 + 99 * opts.sgd_ratio) / 100;%here use the diminish sgd ratio
    d = -f.grad(x,opts);
    if norm(d) < opts.gm.sgd_epsilon
        opts.sgd_ratio = 1;
    end
    % Calculate Alpha 
    if strcmp(opts.gm.step_size_method, "exact")
        alpha = exact_line_search(f,x,opts);
    elseif strcmp(opts.gm.step_size_method, "armijo")
        alpha = armijo_line_search(f,x,d,opts);
    elseif strcmp(opts.gm.step_size_method, "diminishing")
        alpha = diminishing_step_size(k, opts.diminishing.p);
    end
    
    % Calculate next x
    old_x = x;
    x = x + alpha * d;
    new_x = x;
    
    % Plot trace
    if opts.gm.plot
        plot([old_x(1) new_x(1)], [old_x(2), new_x(2)]);
    end
    
    % Calculate Gradient 
    grad = f.grad(x,opts);
    
    % Check if stopping crkia is satisfied
    ng = norm(grad);
    
    % Add new element to arrays
    ks(k) = k;
    ngs(k) = ng;
%     norms(k) = norm(x - [1;1]);
    
    % test accuracy
    [CR_train,CR_test] = train_test_accuracy(x);
    train_accs(k) = CR_train;
    test_accs(k) = CR_test;
    
    if opts.gm.print
        obj_val   = f.obj(x,opts);
        fprintf('k=[%5i] ; obj_val=%1.6f ; ng=%1.4e ; alpha=%1.2f ; train_acc=%1.4f ; test_acc=%1.4f\n',k,obj_val,ng,alpha,CR_train, CR_test);
    end
    
    if ng <= opts.gm.tol
        break
    end
    
end
    