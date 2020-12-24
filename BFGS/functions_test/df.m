function dfx = df(x)
%DF(X) 此处显示有关此函数的摘要
%   此处显示详细说明
x1 = x(1);x2 = x(2);
dfx = 2 * f1(x) * df1(x) + 2 * f2(x) * df2(x);
end

