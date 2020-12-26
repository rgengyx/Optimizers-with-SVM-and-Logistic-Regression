syms x [1 2]
f.syms.obj = (3 + x(1) + ((1-x(2))*x(2)-2)*x(2))^2 + (3 + x(1) + (x(2)-3)*x(2))^2;
f.syms.grad = gradient(f.syms.obj);
f.syms.hessian = jacobian(f.syms.grad);

sol = solve(f.syms.grad, x);
sol.x1;
sol.x2;

f.hessian = matlabFunction(f.syms.hessian);
eval(f.hessian(sol.x1(1,1),sol.x2(1,1)))
eval(f.hessian(sol.x1(2,1),sol.x2(2,1)))
eval(f.hessian(sol.x1(3,1),sol.x2(3,1)))
eval(f.hessian(sol.x1(4,1),sol.x2(4,1)))
eval(f.hessian(sol.x1(5,1),sol.x2(5,1)))

eig(eval(f.hessian(sol.x1(1,1),sol.x2(1,1))))
eig(eval(f.hessian(sol.x1(2,1),sol.x2(2,1))))
eig(eval(f.hessian(sol.x1(3,1),sol.x2(3,1))))
eig(eval(f.hessian(sol.x1(4,1),sol.x2(4,1))))
eig(eval(f.hessian(sol.x1(5,1),sol.x2(5,1))))

