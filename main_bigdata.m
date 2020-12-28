clear

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

% load("bigdata/phishing/phishing_train.mat");
% load("bigdata/phishing/phishing_train_label.mat");

% Sample
opts.sample.m = size(A,1);
opts = config(opts);
opts.x0 = zeros(size(A,2) + 1, 1);

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs

method_cmp_list = {"lbfgs"};
x_list = {};k_list = {};ngs_list = {};
for i = 1:length(method_cmp_list)%use tic toc here to measure the time consume
    tic
    [x_list{i},k_list{i},ngs_list{i}] = run("svm_sparse",method_cmp_list{i},opts);
    toc
end

% x = run("svm_sparse","gm",opts);

%%%%%%%%
% test %
%%%%%%%%

% ac_list = [];
% for i = 1:length(x_list)%here use global, no return value, maybe change further
%     [CR_train,CR_test] = train_test_accuracy(x_list{i});
%     ac_list(:,i) = [CR_train,CR_test]';
% end

%%%%%%%%%%%%%
% Visualize %
%%%%%%%%%%%%%

for i = 1:length(ngs_list)
    plot(log(ngs_list{i}));
    hold on;
end
legend(method_cmp_list);
