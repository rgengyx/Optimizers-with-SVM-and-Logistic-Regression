function plot_scatter(data1, label1)
    %visual the points(little slow as use for loop)
    l = size(label1);
    l = l(2);
    parfor i = 1:l
        if label1(i) == 1
            scatter(data1(1,i),data1(2,i),'r*');
            hold on;
        else
            scatter(data1(1,i),data1(2,i),'b*');
            hold on;
        end
    end
end