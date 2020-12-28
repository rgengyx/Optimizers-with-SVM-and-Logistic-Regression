function f = svm()

    %%%%%%%%%%%%%%%%%%%%
    % Function Options %
    %%%%%%%%%%%%%%%%%%%%

    f.obj = @svm_obj;
    f.grad = @svm_grad;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Main Functions Declarations %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function fx = svm_obj(xy,opts)
        x = xy(1:end-1,1); y = xy(end);
        lambda = opts.svm.lambda;
        fx = lambda / 2 * norm(x)^2 + penalty(x,y,opts);
    end

    function g = svm_grad(xy,opts)
        x = xy(1:end-1,1); y = xy(end);
        lambda = opts.svm.lambda;
        gx = lambda * x + dx_penalty(x,y,opts);
        gy = dy_penalty(x,y,opts);
        g = [gx;gy];
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Utility Functions Declarations %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function phi_plus_t = phi_plus(a,b,x,y,opts)
        delta = opts.svm.delta;
        t = 1 - b * (a' * x + y);
        if t <= delta
            phi_plus_t = 1/(2 * delta) * (max(0, t))^2;
        else
            phi_plus_t = t - delta / 2;
        end
    end

    function penalty = penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        penalty = 0;
        m = 2;
        for i=1:m
            a = data(:,i);
            b = label(i);
            penalty = penalty + phi_plus(a,b,x,y,opts);
        end
    end

    function penalty = dx_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = 2;
        penalty = 0;
        delta = opts.svm.delta;
        for i=1:m
            a = data(:,i);
            b = label(i);
            t = 1 - b * (a' * x + y);
            if t > 0 && t <= delta
                penalty = penalty + 1/delta * (-1) * b * a;
            elseif t <= 0            
                penalty = penalty + 0;
            else
                penalty = penalty + (-1) * b * a;
            end
        end
    end

    function penalty = dy_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = 2;
        penalty = 0;
        delta = opts.svm.delta;
        for i=1:m
            a = data(:,i);
            b = label(i);
            t = 1 - b * (a' * x + y);
            if t > 0 && t <= delta
                penalty = penalty + (-1) * 1/delta * b;
            elseif t <= 0
                penalty = penalty + 0;
            else
                penalty = penalty + (-1) * b;
            end
        end
    end

end