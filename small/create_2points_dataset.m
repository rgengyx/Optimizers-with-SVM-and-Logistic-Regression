function [data,label] = create_2points_dataset(p1,p2)

%create the small dataset

%hyperparameter and initial points setting
c1 = [1,1]';sigma1 = 0.4;m1 = 200;
c2 = [0,0]';sigma2 = 0.3;m2 = 400;

c1 = p1.c;sigma1 = p1.sigma;m1 = p1.m;
c2 = p2.c;sigma2 = p2.sigma;m2 = p2.m;

%points creating (with broadcasting)
ai = c1 + normrnd(0,sigma1,2,m1);%class 1 points
bi = ones(1,m1);%class 1 label: 1

aj = c2 + normrnd(0,sigma2,2,m2);%class 2 points
bj = -ones(1,m2);%class 2 label: -1

data = [ai,aj];
label = [bi,bj];
end

