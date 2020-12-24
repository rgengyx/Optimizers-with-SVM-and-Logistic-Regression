function [alpha] = exact_line_search(f,x,opts)

syms alpha_syms;
phi.alpha = matlabFunction(f.obj(x - alpha_syms * f.grad(x)));
alpha = ausection(phi.alpha,opts.exact.xl,opts.exact.xr,opts.exact);

end