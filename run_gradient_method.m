% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
% Global Seed Settings
rng("default");

%%%%%%%%%%%%%
% Load data %
%%%%%%%%%%%%%
global data1;global label1;
load("small/small_dataset_mod.mat");
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

% GM
opts.gm.maxit = 100;
opts.gm.tol = 1e-4;
opts.gm.display = true;
opts.gm.step_size_method = "armijo";
opts.gm.plot = false;
opts.gm.print = true;

% Armijo
opts.armijo.maxit = 100;
opts.armijo.s = 1;
opts.armijo.sigma = 0.5;
opts.armijo.gamma = 0.1;

% Sample
opts.sample.m = length(data1);


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
%f = svm();
f = logistic_regression();

x0 = [0;0;0];
[x,ks,ngs] = gradient_method(f,x0,opts);


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

plot_scatter(data2,label2);
hold on

x1 = -3:0.01:3;
x2 = calculate_x2(x,x1);

plot(x1,x2);


%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

function x2 = calculate_x2(x, x1)
    x2 = (-x1 * x(1) - x(3))/x(2); 
end

