function [x_end,count] = L_BFGS(f,x0,opts)
%BFGS 此处显示有关此函数的摘要
%   此处显示详细说明
addpath('D:\desktop2\new start learning\cuhksz learning\optimization-MDS6106\project\project_git\MDS6106_Project\search')
%addpath('functions_test\')
%hyperparameter setting
%just use the opts
limit_step = opts.limit_step;

count = 0;
x_now = x0;

%global s_buffer;global y_buffer;global alpha_buffer;
while(count < opts.maxit && norm(f.grad(x_now)) > opts.epsilon)
    %mimit the two-recursion loop, modify it with the former m steps
    %when enter a loop, x_now is known, former m steps buffer is known
    q = -f.grad(x_now);
    
    %recursion 1
    buffer_end = min(count,limit_step);
    if buffer_end == 0 %the first step
        H0_now_const = 0.5;
    else
        H0_now_const = (s_buffer{buffer_end}' * y_buffer{buffer_end}) / (y_buffer{buffer_end}' * y_buffer{buffer_end});
    end
    
    for i = 1:buffer_end
        i_new = buffer_end + 1 - i;%do the up-side down
        rou_now = 1 / (s_buffer{i_new}' * y_buffer{i_new});
        alpha_buffer{i_new} = rou_now * s_buffer{i_new}' * q;%store the alpha
        q = q - alpha_buffer{i_new} * y_buffer{i_new};
    end
    
    %set r for recursion mid
    r = H0_now_const * q;
    
    %recursion 2
    for i = 1:buffer_end
        rou_now = 1 / (s_buffer{i}' * y_buffer{i});
        beta_now = rou_now * y_buffer{i}' * r;
        r = r + (alpha_buffer{i} - beta_now) * s_buffer{i};
    end
    
    %calculate the x_next
    d_now = r;
    alpha = armijo_line_search(f,x_now,d_now,opts);
    x_next = x_now + alpha * d_now;
    
    %renew the buffer storage
    if buffer_end < limit_step %if not reach buffer end, just append
        buffer_end = buffer_end + 1;
        s_buffer{buffer_end} = x_next - x_now;
        y_buffer{buffer_end} = f.grad(x_next) - f.grad(x_now);
    else
        for i = 1:buffer_end - 1 %if reach buffer, shift the buffer
            s_buffer{i} = s_buffer{i + 1};
            y_buffer{i} = y_buffer{i + 1};
        end
        s_buffer{buffer_end} = x_next - x_now;
        y_buffer{buffer_end} = f.grad(x_next) - f.grad(x_now);
    end
    
    %renew the x_now
    x_now = x_next;
    count = count + 1;
end
x_end = x_now;
end

