% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));

addpath('D:\desktop2\new start learning\cuhksz learning\optimization-MDS6106\project\datasets\datasets')

% Global Seed Settings
rng("default");

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;
%load("small/small_dataset_sample.mat");
load('D:\desktop2\new start learning\cuhksz learning\optimization-MDS6106\project\datasets\datasets\mushrooms\mushrooms_train.mat')
load('D:\desktop2\new start learning\cuhksz learning\optimization-MDS6106\project\datasets\datasets\mushrooms\mushrooms_train_label.mat')
%I used full matrix here, but should be modified for big dimension
data1 = full(A');
label1 = full(b');
train_ratio = 0.8;len_data = length(data1);
rand_index = randperm(len_data,len_data);
train_index = rand_index(1:floor(len_data * train_ratio));
test_index = rand_index((floor(len_data * train_ratio) + 1):len_data);
%split dataset

data2 = data1(:,test_index);label2 = label1(test_index);
data1 = data1(:,train_index);label1 = label1(train_index);

%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

% BFGS
opts.lbfgs.epsilon = 1e-6;
opts.lbfgs.H_epsilon = 1e-14;
opts.lbfgs.rou = 1;
opts.lbfgs.maxit = 400;
opts.lbfgs.limit_step = 10;%range [5,25]

% Armijo
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;


% Sample
opts.sample.dim = size(data1);
opts.sample.m = opts.sample.dim(2);


%%%%%%%%%%%%%%%%%%%%%%
% Parameters Options %
%%%%%%%%%%%%%%%%%%%%%%

% SVM
opts.svm.lambda = 1/opts.sample.m;
opts.svm.delta = 1e-4;

% Logistic Regression
opts.logr.lambda = 0.1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% f = svm();
f = logistic_regression();
%f = bfgs();

x0 = zeros(opts.sample.dim(1) + 1,1);

tic
[x,k] = L_BFGS(f,x0,opts);
toc
%%%%%%%%%%%%%%%%%
%     test      %
%%%%%%%%%%%%%%%%%
opt1.paras = x;
%bar for SVM problem
opt1.bar = 0;opt1.label_bar = 0;
%bar for Logistic problem
opt1.bar = 0.5;opt1.label_bar = 0.5;

CR_train = cal_CR(data1,label1,opt1);
CR_test = cal_CR(data2,label2,opt1);

%%%%%%%%%%%%%%%%%
% Visualization %
%%%%%%%%%%%%%%%%%
%{
plot_scatter(data2,label2);
hold on

x1 = -3:0.01:3;
x2 = calculate_x2(x,x1);

plot(x1,x2);
%}

%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

function x2 = calculate_x2(x, x1)
    x2 = (-x1 * x(1) - x(3))/x(2); 
end

