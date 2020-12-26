%create the small dataset

%hyperparameter and initial points setting
p1.c = [0,0]';p1.sigma = 0.4;p1.m = 200;
p2.c = [1,1]';p2.sigma = 0.5;p2.m = 300;

%dataset create
[data1,label1] = create_2points_dataset(p1,p2);

%data save
save('small_dataset','data1','label1')
