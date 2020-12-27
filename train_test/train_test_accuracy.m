function train_test_accuracy(x, train_split)

    global data1;global label1;global data2;global label2;
    train_ratio = train_split;len_data = length(data1);
    rand_index = randperm(len_data,len_data);
    train_index = rand_index(1:floor(len_data * train_ratio));
    test_index = rand_index((floor(len_data * train_ratio) + 1):len_data);
    %split dataset
    data2 = data1(:,test_index);label2 = label1(test_index);
    data1 = data1(:,train_index);label1 = label1(train_index);


    opt1.paras = x;
    %bar for SVM problem
    opt1.bar = 0;opt1.label_bar = 0;
    %bar for Logistic problem
    opt1.bar = 0.5;opt1.label_bar = 0.5;
    CR_train = cal_CR(data1,label1,opt1);
    CR_test = cal_CR(data2,label2,opt1);

    fprintf("Train Accuracy = %f, Test Accuracy = %f", CR_train, CR_test);

end