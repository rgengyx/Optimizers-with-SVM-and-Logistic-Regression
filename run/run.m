function x = run(func, method, opts)
    
% import
addpath(genpath('run'));

    if method == "gm"
        x = run_gradient_method(func, opts)
    elseif method == "agm"
        x = run_accelerated_gradient_method(func,opts)
    elseif method == "bfgs"
        x = run_bfgs(func,opts)
    elseif method == "lbfgs"
        x = run_L_bfgs(func,opts)
    end

end