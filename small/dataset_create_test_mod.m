%create the small dataset

%hyperparameter and initial points setting
p1.c = [1.5,0.5]';p1.sigma = 0.4;p1.m = 200;
p2.c = [1,1]';p2.sigma = 0.5;p2.m = 300;

%dataset create
[data1,label1] = create_2points_dataset_mod(p1,p2);

%data save
save('small_dataset_mod','data1','label1')


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
