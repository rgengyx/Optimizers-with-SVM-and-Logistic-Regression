function f = logistic_regression_sgd()

    %%%%%%%%%%%%%%%%%%%%
    % Function Options %
    %%%%%%%%%%%%%%%%%%%%
    
    f.obj = @logr_obj;
    f.grad = @logr_grad;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Main Functions Declarations %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function fx = logr_obj(xy, opts)
        x = xy(1:end-1,1); y = xy(end);
        lambda = opts.svm.lambda;
        fx = lambda / 2 * norm(x)^2 + penalty(x,y,opts);
    end

    function g = logr_grad(xy,opts)
        x = xy(1:end-1,1); y = xy(end);
        lambda = opts.svm.lambda;
        gx = lambda * x + dx_penalty(x,y,opts);
        gy = dy_penalty(x,y,opts);
        g = [gx;gy];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Utility Functions Declarations %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    function penalty = penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        penalty = 0;
        summation = 0;
        m = opts.sample.m;
        for i=1:m
            a = data(:,i);
            b = label(i);
            t = 1 + exp(-b * (a' * x + y));
            summand = log(t);
            summation = summation + summand;
        end
        penalty = 1/m * summation;
    end

    function penalty = dx_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = opts.sample.m;
        penalty = 0;
        summation = 0;
        for i=1:m
            a = data(:,i);
            b = label(i);
            summand = (b*a*exp(-b * (a' * x + y))) / (1+exp(-b * (a' * x + y)));
            summation = summation + summand;
        end
        penalty = -1/m * summation;
    end

    function penalty = dy_penalty(x,y,opts)
        global data1 label1;
        data = data1; label = label1;
        m = opts.sample.m;
        penalty = 0;
        summation = 0;
        for i=1:m
            a = data(:,i);
            b = label(i);
            summand = (b*exp(-b * (a' * x + y))) / (1+exp(-b * (a' * x + y)));
            summation = summation + summand;
        end
        penalty = -1/m * summation;
    end
end