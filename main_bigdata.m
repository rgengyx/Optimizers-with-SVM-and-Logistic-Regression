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
global A b;

load("bigdata/mushrooms/mushrooms_train.mat");
load("bigdata/mushrooms/mushrooms_train_label.mat");

% Sample
opts.sample.m = size(A,1);
opts = config(opts);
opts.x0 = zeros(size(A,2) + 1, 1);

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs

x = run("svm_sparse","lbfgs",opts);

%%%%%%%%
% test %
%%%%%%%%

% train_test_accuracy(x, 0.8);

%%%%%%%%%%%%%
% Visualize %
%%%%%%%%%%%%%

% visualize(x, data2, label2);

