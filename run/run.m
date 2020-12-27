function [x,count,ngs] = run(func, method, opts)
% import
addpath(genpath('run'));

    if method == "gm"
        [x,count,ngs] = run_gradient_method(func, opts);
    elseif method == "agm"
        [x,count,ngs] = run_accelerated_gradient_method(func,opts);
    elseif method == "bfgs"
        [x,count,ngs] = run_bfgs(func,opts);
    elseif method == "lbfgs"
        [x,count,ngs] = run_L_bfgs(func,opts);
    end

end