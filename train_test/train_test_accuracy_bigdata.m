function [CR_train,CR_test] = train_test_accuracy_bigdata(x)
    global data1;global label1;global data2;global label2;

    opt1.paras = x;
    %bar for SVM problem
    opt1.bar = 0;opt1.label_bar = 0;
    %bar for Logistic problem
    opt1.bar = 0.5;opt1.label_bar = 0.5;
    CR_train = cal_CR_bigdata(data1,label1,opt1);
    CR_test = cal_CR_bigdata(data2,label2,opt1);

%     fprintf("Train Accuracy = %f, Test Accuracy = %f", CR_train, CR_test);

end