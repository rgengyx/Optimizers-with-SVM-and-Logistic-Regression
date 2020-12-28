function [x,k,ngs, train_accs, test_accs] = run_accelerated_gradient_method_batch(func, opts)

% Add folder to path
addpath(genpath('method'));
addpath(genpath('function'));
addpath(genpath('visualization'));
addpath(genpath('search'));
addpath(genpath('test'));
% Global Seed Settings
rng("default");


%%%%%%%%%%%%%%%%%%%
% Method Options %
%%%%%%%%%%%%%%%%%%%

opts.agm.beta = @beta;
opts.agm.step_size = step_size(opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global data1;global label1;
data1_ori = data1;label1_ori = label1; %save the original data and label
tol_ori = opts.agm.tol;%save the original epsilon
sizes = size(label1_ori);
ori_m = sizes(2);

%batch settings
rand_index = randperm(opts.sample.m,opts.sample.m);

%set the batch size here, maybe can change in main run
batch_size = opts.bfgs.batch_size;batch_count = ceil(ori_m / batch_size);


x = opts.x0;k = 0;ngs = [];train_accs=[];test_accs=[];
for i = 1:batch_count %do with the batch
    if issparse(data1) == 1
        data1 = data1_ori(rand_index((i-1) * batch_size + 1:min(i * batch_size,ori_m)),:);
        label1 = label1_ori(rand_index((i-1) * batch_size + 1:min(i * batch_size,ori_m)));
    else
        data1 = data1_ori(:,rand_index((i-1) * batch_size + 1:min(i * batch_size,ori_m)));
        label1 = label1_ori(rand_index((i-1) * batch_size + 1:min(i * batch_size,ori_m)));
    end
    
    %renew the m as needs
    sizes = size(label1);
    opts.sample.m = sizes(2);
    
    %renew the epsilon as batch is more tolarent
    opts.agm.tol = opts.agm.batch_epsilon;
    
    %use new batch data to run new f
    if func == "svm"
        f = svm();
    elseif func == "svm_sparse"
        f = svm_sparse();
    elseif func == "logr"
        f = logr();
    elseif func == "logr_sparse"
        f = logr_sparse();
    elseif func == "logr_sgd"
        f = logr_sgd();
    elseif func == "logr_sgd_sparse"
        f = logr_sgd_sparse();
    end
    
    %use the former batch result for next train
    x0 = x;
    [x,k_new,ngs_new,train_accs_new, test_accs_new] = agm_unknown(f,x0,opts);
    %[x,k_new,ngs_new,train_accs, test_accs] = agm_known(f,x0,opts);

    %store the res
    k = k + k_new;
    ngs = [ngs,ngs_new];
    train_accs = [train_accs, train_accs_new];
    test_accs = [test_accs, test_accs_new];
end

%finally use the whole dataset for final descent
data1 = data1_ori;label1 = label1_ori;
%renew the m as needs
sizes = size(label1);opts.sample.m = sizes(2);
%renew the epsilon as before
opts.agm.tol = tol_ori;

%use the former batch result for final train
x0 = x;
[x,k_new,ngs_new,train_accs_new, test_accs_new] = agm_unknown(f,x0,opts);
%[x,k_new,ngs_new] = agm_known(f,x0,opts);

%store the res
k = k + k_new;
ngs = [ngs,ngs_new];
train_accs = [train_accs, train_accs_new];
test_accs = [test_accs, test_accs_new];
%%%%%%%%%%%%%%%%%%%%%
% Utility Functions %
%%%%%%%%%%%%%%%%%%%%%

% Define Extrapolation parameter beta
function [prev_t, beta1] = beta(prev_t)
    t = (1/2)*(1 + sqrt(1+4*prev_t^2));
    beta1 = (prev_t - 1) / t;
    prev_t = t;
end

% Step size
function step_size1 = step_size(opts)
    L = opts.agm.L;
    step_size1 = 1 / L;
end

end