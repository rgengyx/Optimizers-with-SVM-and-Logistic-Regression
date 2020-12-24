function x_end = L_BFGS(f,x0,opts)
%BFGS 此处显示有关此函数的摘要
%   此处显示详细说明
addpath('D:\desktop2\new start learning\cuhksz learning\optimization-MDS6106\project\project_git\MDS6106_Project\search')
addpath('functions_test\')
%hyperparameter setting
%just use the opts
count = 0;

%build H0 with the dimension of x0
n = size(x0);n = n(1);
H_now = eye(n) * opts.rou;
df = f.grad;
%set initial x_now
x_now = x0;

while(count < opts.maxit)
    d_now = -H_now * df(x_now);
    alpha = armijo_line_search(f,x_now,d_now,opts);
    x_next = x_now + alpha * d_now;

    if(df(x_next) <= opts.epsilon)
        break;
    end
    
    %build sk and yk
    s_now = x_next - x_now;y_now = df(x_next) - df(x_now);
    
    %Hk+1 setting
    if(s_now' * y_now < 0)%not good Hk+1, hold original
        H_next = H_now;
    else
        add1 = ((s_now - H_now * y_now) * s_now' + s_now * (s_now - H_now * y_now)') / (s_now' * y_now);
        add2 = (s_now - H_now * y_now)' * y_now * (s_now * s_now') / (s_now' * y_now)^2;
               
        H_next = H_now + add1 - add2;
    end
    
    %renew the x_now and H_now
    x_now = x_next;
    H_now = H_next;
    count = count + 1;
end

x_end = x_now;
end

