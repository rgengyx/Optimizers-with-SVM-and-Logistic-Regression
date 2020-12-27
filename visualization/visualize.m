function visualize(x, data, label)

    %%%%%%%%%%%%%%%%%
    % Visualization %
    %%%%%%%%%%%%%%%%%

    plot_scatter(data,label);
    hold on
    x1 = -3:0.01:3;
    x2 = calculate_x2(x,x1);
    plot(x1,x2);


    %%%%%%%%%%%%%%%%%%%%%
    % Utility Functions %
    %%%%%%%%%%%%%%%%%%%%%

    function x2 = calculate_x2(x, x1)
        x2 = (-x1 * x(1) - x(3))/x(2); 
    end

end