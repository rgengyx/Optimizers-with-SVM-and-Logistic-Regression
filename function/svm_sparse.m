function f = svm_sparse()

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

    function phi_plus_t = phi_plus(ai,bi,x,y,opts)
        delta = opts.svm.delta;
        t = 1 - bi * (ai' * x + y);
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
        m = size(data1,1);
        for i=1:m
            ai = data(i,:)';
            bi = label(i);
            penalty = penalty + phi_plus(ai,bi,x,y,opts);
        end
    end

    function penalty = dx_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = size(data1,1);
        penalty = 0;
        delta = opts.svm.delta;
        for i=1:m
            ai = data(i,:)';
            bi = label(i);
            t = 1 - bi * (ai' * x + y);
            if t > 0 && t <= delta
                penalty = penalty + 1/delta * (-1) * bi * ai;
            elseif t <= 0            
                penalty = penalty + 0;
            else
                penalty = penalty + (-1) * bi * ai;
            end
        end
    end

    function penalty = dy_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = size(data1,1);
        penalty = 0;
        delta = opts.svm.delta;
        for i=1:m
            ai = data(i,:)';
            bi = label(i);
            t = 1 - bi * (ai' * x + y);
            if t > 0 && t <= delta
                penalty = penalty + (-1) * 1/delta * bi;
            elseif t <= 0
                penalty = penalty + 0;
            else
                penalty = penalty + (-1) * bi;
            end
        end
    end

end