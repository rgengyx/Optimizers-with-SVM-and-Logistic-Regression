function [alpha] = armijo_line_search(f,x,d,opts)

alpha = opts.armijo.s;

while f.obj(x + alpha * d, opts) > f.obj(x, opts) + opts.armijo.gamma * alpha * f.grad(x, opts)'* d
    alpha = opts.armijo.sigma * alpha;
end

end