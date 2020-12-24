function [alpha] = armijo_line_search(f,x,d,opts)

alpha = opts.armijo.s;
f.obj(x + alpha * d);
x + alpha * d;
f.obj(x);
x;
opts.armijo.gamma * alpha * f.grad(x)'* d;
while f.obj(x + alpha * d) > f.obj(x) + opts.armijo.gamma * alpha * f.grad(x)'* d
    alpha = opts.armijo.sigma * alpha;
    
    
end

end