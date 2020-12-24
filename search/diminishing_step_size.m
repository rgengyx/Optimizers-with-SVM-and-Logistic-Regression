function [alpha] = diminishing_step_size(k,p)

alpha = 0.01/log(k+p);
    
end