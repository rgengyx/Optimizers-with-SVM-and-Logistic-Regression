% Add folder to path
addpath(genpath('method'));
addpath(genpath('run'));
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

% Sample
opts.sample.m = length(data1);
opts = config(opts);

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs

x = run_model("svm","gm",opts);

%%%%%%%%
% test %
%%%%%%%%

train_test_accuracy(x, 0.8);

%%%%%%%%%%%%%
% Visualize %
%%%%%%%%%%%%%

visualize(x, data2, label2);