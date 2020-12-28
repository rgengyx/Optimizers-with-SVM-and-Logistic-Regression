function [x,count,ngs] = run(func, method, opts)
% import
addpath(genpath('run'));

    if method == "gm"
        [x,count,ngs] = run_gradient_method(func, opts);
    elseif method == "gm_batch"
        [x,count,ngs] = run_gradient_method_batch(func, opts);
    elseif method == "gm_sgd"
        [x,count,ngs] = run_gradient_method(func+"_sgd", opts);
    elseif method == "agm"
        [x,count,ngs] = run_accelerated_gradient_method(func,opts);
    elseif method == "agm_batch"
        [x,count,ngs] = run_accelerated_gradient_method_batch(func,opts);
    elseif method == "agm_sgd"
        [x,count,ngs] = run_accelerated_gradient_method(func+"_sgd",opts);
    elseif method == "bfgs"
        [x,count,ngs] = run_bfgs(func,opts);
    elseif method == "bfgs_batch"
        [x,count,ngs] = run_bfgs_batch(func,opts);
    elseif method == "bfgs_sgd"
        [x,count,ngs] = run_bfgs(func + "_sgd",opts);%use the original bfgs, while change the function
    elseif method == "lbfgs"
        [x,count,ngs] = run_L_bfgs(func,opts);
    elseif method == "lbfgs_batch"
        [x,count,ngs] = run_L_bfgs_batch(func, opts);
    elseif method == "lbfgs_sgd"
        [x,count,ngs] = run_L_bfgs(func + "_sgd",opts);%use the original bfgs, while change the function
    end

end