% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
addpath(genpath('train_test'));

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;global data2;global label2;

load("small/small_dataset_sample.mat");

train_ratio = 0.8;len_data = length(data1);
rand_index = randperm(len_data,len_data);
train_index = rand_index(1:floor(len_data * train_ratio));
test_index = rand_index((floor(len_data * train_ratio) + 1):len_data);
%split dataset
data2 = data1(:,test_index);label2 = label1(test_index);
data1 = data1(:,train_index);label1 = label1(train_index);

% Sample
opts.sample.m = length(data1);
opts = config(opts);

%%%%%%%
% Run %
%%%%%%%

x = run("svm","gm",opts);

%%%%%%%%
% test %
%%%%%%%%

train_test_accuracy(x);

%%%%%%%%%%%%%
% Visualize %
%%%%%%%%%%%%%

visualize(x, data2, label2);