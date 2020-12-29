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

dataset = "mushrooms";

load("bigdata/"+dataset+"/"+dataset+"_train.mat");
load("bigdata/"+dataset+"/"+dataset+"_train_label.mat");

data1 = A;label1 = b;

% Sample
opts.sample.m = size(data1,1);
opts = config(opts);
opts.x0 = zeros(size(data1,2) + 1, 1);

%split data, data1 for train, data2 for test
rand_index = randperm(opts.sample.m,opts.sample.m);split_index = floor(opts.sample.m * opts.split_ratio);
data2 = data1(rand_index(1:split_index),:);
label2 = label1(rand_index(1:split_index));

data1 = data1(rand_index(split_index+1:end),:);
label1 = label1(rand_index(split_index+1:end));

%%%%%%%
% Run %
%%%%%%%

% svm, logr
% gm, agm, bfgs, lbfgs
% gm_batch,agm_batch,bfgs_batch,lbfgs_batch
% gm_sgd
% gm_sgd_batch

method_cmp_list = {"lbfgs"};
lambdas = {0.05, 0.1,0.2,0.5,0.8};

for i = 1:length(lambdas)%use tic toc here to measure the time consume
    tic
    opts.logr.lambda = lambdas{i};
    [x_list{i},k_list{i},ngs_list{i},train_accs_list{i},test_accs_list{i}] = run("logr_sparse",method_cmp_list{1},opts);
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
legend({"0.05", "0.1", "0.2", "0.5", "0.8"});

% Accuracy Plot
figure('Name','Testing Accuracy');
for i = 1:length(lambdas)
%     plot(train_accs_list{i});
%     hold on;
    plot(test_accs_list{i});
    hold on;
end

legend({"0.05", "0.1", "0.2", "0.5", "0.8"});
