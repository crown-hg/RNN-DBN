tic;
clear;
clc;
%% 基本参数
% 前面几个参数是用于我自己写的BP，然而效果不好，就没有用
% lambda = 0.00001;      %weight decay parameter权重衰减参数
% eta =0.6;              %学习率
% eb = 0.1 ;             %误差容限
% maxIter = 2000;        %最大迭代次数 
% mc = 0.8;              %动量因子

addpath data/
addpath RBM/
load('pemsd05_2013_month123_day89_link100.mat');

%% 使用英国数据
%分daytime和nighttime
% daytime=zeros(96,1);
% daytime(21:84)=1;
% traindaytime=repmat(daytime,92,1);
% testdaytime=repmat(daytime,29,1);
% traindata = train.data(traindaytime==1,:);
% trainlabels = train.labels(traindaytime==1,:);
% testdata = test.data(testdaytime==1,:);
% testlabels = test.labels(testdaytime==1,:);


% PeMS数据
numday=88;
daytimesize=96;
numlink=100;
numperiod=3;

[data,labels] = createPemsTraindata(daydata, numday,daytimesize,numlink,numperiod);
numtrain = 5760;
numtest = 2688;
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);


%% 通用


weekuse=0; %0是weekday工作日，1是weekend双休日
time = 1;%1为daytime，0为nighttime
hidelayer = [300];

% 分week和weekend
weektrain=zeros(size(traindata,1),1);
for i=1:size(traindata,1)
   if mod(ceil(i/96)+3,7)==6||mod(ceil(i/96)+3,7)==0
      weektrain(i)=1;
   end
end
weektest=zeros(size(testdata,1),1);
for i=1:size(testdata,1)
   if mod(ceil(i/96),7)==6||mod(ceil(i/96),7)==0
      weektest(i)=1;
   end
end


traindata=traindata(weektrain==weekuse,:);
trainlabels=trainlabels(weektrain==weekuse,:);
testdata=testdata(weektest==weekuse,:);
testlabels=testlabels(weektest==weekuse,:);

% 分白天晚上
daytime=zeros(96,1);
daytime(21:84)=1; %早上6点到晚上8点
traindaytime=repmat(daytime,size(traindata,1)/96,1);
testdaytime=repmat(daytime,size(testdata,1)/96,1);

traindata = traindata(traindaytime==time,:);
trainlabels = trainlabels(traindaytime==time,:);
testdata = testdata(testdaytime==time,:);
testlabels = testlabels(testdaytime==time,:);

numtest = size(testdata,1);

numhide = size(hidelayer,2);
op1.verbose=true;
op1.maxepoch=50;
op2.verbose=true;
op2.maxepoch=50;
op3.verbose=true;
op4.maxepoch=50;

models=dbnFit(traindata,hidelayer,trainlabels,op1,op2,op3); %训练
func = @logbpcost;
m = bpFine(models, traindata, trainlabels, func);
% m=bp(eb,mc,eta,maxIter,lambda, models, traindata, trainlabels);
% 这是我写的BP，效果不如牛顿法

% testdata = traindata;
% testlabels = trainlabels;
a = runnet(testdata, m.W, m.b, numhide);
h = a{numhide+2};
me = sum(sum(abs(h-testlabels)))/sum(sum(testlabels));
dp=mapminmax('reverse',h,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;
rdata=mapminmax('reverse',traindata,ps);
MRE = sum(sum(abs(dp-dr)./dr))/(numlink*numtest);
MAE = sum(sum(abs(dp-dr)))/(numlink*numtest);
RMSE = sqrt(sum(sum((dp-dr).^2))/(numlink*numtest));
figure(1);
plot(1:numlink,sum(abs(dp-dr)./dr)/numtest,'*r');
figure(2);
plot(1:numlink,sum(abs(dp-dr))/numtest,'*b');
toc;