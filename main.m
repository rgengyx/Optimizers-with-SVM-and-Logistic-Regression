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
global A; global b;

load("small/small_dataset_sample.mat");
% load("bigdata/mushrooms/mushrooms_train.mat");
% load("bigdata/mushrooms/mushrooms_train_label.mat");

% Sample
sizes = size(data1);
opts.sample.m = sizes(2);%the count of sample
opts = config(opts);

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs
% agm_sgd, bfgs_sgd, lbfgs_sgd

%initial point set
opts.x0 = [0,0,0]';
method_cmp_list = {"gm","gm_sgd"};
x_list = {};k_list = {};ngs_list = {};
for i = 1:length(method_cmp_list)%use tic toc here to measure the time consume
    tic
    [x_list{i},k_list{i},ngs_list{i}] = run("logr",method_cmp_list{i},opts);
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
for i = 1:length(ngs_list)
    plot(log(ngs_list{i}));
    hold on;
end
legend(method_cmp_list);
