function [x, k, ngs, train_accs, test_accs] = run(func, method, opts)
% import
addpath(genpath('run'));

    if method == "gm"
        [x, k, ngs, train_accs, test_accs] = run_gradient_method(func, opts);
    elseif method == "gm_batch"
        [x, k, ngs, train_accs, test_accs] = run_gradient_method_batch(func, opts);
    elseif method == "gm_sgd"
        [x, k, ngs, train_accs, test_accs] = run_gradient_method(func+"_sgd", opts);
    elseif method == "gm_sgd_batch"
        [x, k, ngs, train_accs, test_accs] = run_gradient_method_batch(func+"_sgd", opts);
    elseif method == "agm"
        [x, k, ngs, train_accs, test_accs] = run_accelerated_gradient_method(func,opts);
    elseif method == "agm_batch"
        [x, k, ngs, train_accs, test_accs] = run_accelerated_gradient_method_batch(func,opts);
    elseif method == "agm_sgd_batch"
        [x, k, ngs, train_accs, test_accs] = run_accelerated_gradient_method_batch(func + "_sgd",opts);
    elseif method == "agm_sgd"
        [x, k, ngs, train_accs, test_accs] = run_accelerated_gradient_method(func+"_sgd",opts);
    elseif method == "bfgs"
        [x, k, ngs, train_accs, test_accs] = run_bfgs(func,opts);
    elseif method == "bfgs_batch"
        [x, k, ngs, train_accs, test_accs] = run_bfgs_batch(func,opts);
    elseif method == "bfgs_sgd"
        [x, k, ngs, train_accs, test_accs] = run_bfgs(func + "_sgd",opts);%use the original bfgs, while change the function
    elseif method == "bfgs_sgd_batch"
        [x, k, ngs, train_accs, test_accs] = run_bfgs_batch(func+"_sgd",opts);
    elseif method == "lbfgs"
        [x, k, ngs, train_accs, test_accs] = run_L_bfgs(func,opts);
    elseif method == "lbfgs_batch"
        [x, k, ngs, train_accs, test_accs] = run_L_bfgs_batch(func, opts);
    elseif method == "lbfgs_sgd"
        [x, k, ngs, train_accs, test_accs] = run_L_bfgs(func + "_sgd",opts);%use the original bfgs, while change the function
    elseif method == "lbfgs_sgd_batch"
        [x, k, ngs, train_accs, test_accs] = run_L_bfgs_batch(func, opts);
    end

end