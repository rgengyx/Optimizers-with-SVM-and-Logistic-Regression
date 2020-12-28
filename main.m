clear

% Add folder to path
addpath(genpath('method'));
addpath(genpath('run'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
addpath(genpath('train_test'));
addpath(genpath('run'));

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;global data2;global label2;

load("small/small_dataset_sample.mat");

% Sample
sizes = size(data1);
opts.sample.m = sizes(2);%the count of sample
opts = config(opts);

%split data, data1 for train, data2 for test
rand_index = randperm(opts.sample.m,opts.sample.m);split_index = floor(opts.sample.m * opts.split_ratio);
data2 = data1(:,rand_index(1:split_index));
label2 = label1(rand_index(1:split_index));

data1 = data1(:,rand_index(split_index+1:end));
label1 = label1(rand_index(split_index+1:end));

sizes = size(data1);
opts.sample.m = sizes(2);%the count of sample

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs
% gm_batch,agm_batch,bfgs_batch,lbfgs_batch
% gm_sgd,agm_sgd, bfgs_sgd, lbfgs_sgd
% gm_sgd_batch,agm_sgd_batch, bfgs_sgd_batch, lbfgs_sgd_batch

%initial point set
opts.x0 = [0,0,0]';
method_cmp_list = {"bfgs_sgd_batch"};
x_list = {};k_list = {};ngs_list = {};train_accs_list = {};test_accs_list = {};
for i = 1:length(method_cmp_list)%use tic toc here to measure the time consume
    tic
    [x_list{i},k_list{i},ngs_list{i},train_accs_list{i},test_accs_list{i}] = run("logr",method_cmp_list{i},opts);
    toc
end
%%%%%%%%
% test %
%%%%%%%%

ac_list = [];
for i = 1:length(x_list)%here use global, no return value, maybe change further
    [CR_train,CR_test] = train_test_accuracy(x_list{i});
    ac_list(:,i) = [CR_train,CR_test]';
end

%%%%%%%%%%%%%
% Visualize %
%%%%%%%%%%%%%

%visualize(x, data2, label2);
% Convergence Plot
figure('Name','Convergence Plot');
for i = 1:length(ngs_list)
    plot(log(ngs_list{i}));
    hold on;
end
legend(method_cmp_list);

% Accuracy Plot
figure('Name','Train Test Accuracy');
for i = 1:length(train_accs_list)
    plot(train_accs_list{i});
    hold on;
end
for i = 1:length(test_accs_list)
    plot(test_accs_list{i});
    hold on;
end
legend({"Training Accuracy", "Test Accuracy"});