function [alpha] = armijo_line_search(f,x,d,opts)

alpha = opts.armijo.s;
<<<<<<< HEAD
f.obj(x + alpha * d);
x + alpha * d;
f.obj(x);
x;
opts.armijo.gamma * alpha * f.grad(x)'* d;
while f.obj(x + alpha * d) > f.obj(x) + opts.armijo.gamma * alpha * f.grad(x)'* d
=======
while f.obj(x + alpha * d, opts) > f.obj(x, opts) + opts.armijo.gamma * alpha * f.grad(x, opts)'* d
>>>>>>> ef7ec97a1a10509ddc168c438572d40c7f550927
    alpha = opts.armijo.sigma * alpha;
    
    
end

end