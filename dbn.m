% function [cost,time,MRE,MAE,RMSE]=dbn(data,labels,numperiod,week,day,hidelayer,bpcost,dbnfit,run,ps)
%% 基本参数
t1=clock;
tic;
daytimesize=96;
week = 1; %1是weekday工作日，0是weekend双休日
day = 1;%1为daytime，0为nighttime
hidelayer = [100];
topfunc=@tanh;
hidefunc=@logsig;

numperiod=4;
addpath data/
dataname=sprintf('new147k%d',numperiod);
load(dataname);

%% PeMS数据
numlink=size(labels,2);
numtrain = 71*96; %前9个月，3月10日和9月17日是去掉的
numtest = 18*96; %第10月
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);

%% 通用
% 分week和weekend
weekflag=ones(size(data,1),1);
for i=1:size(data,1)
   if i<=68*96
       if mod(ceil(i/96)+1,7)==6||mod(ceil(i/96)+1,7)==0
          weekflag(i)=0;
       end
   else
       if i>68*96&&i<=258*96
           if mod(ceil(i/96)+2,7)==6||mod(ceil(i/96)+2,7)==0
              weekflag(i)=0;
           end
       else
           if mod(ceil(i/96)+3,7)==6||mod(ceil(i/96)+3,7)==0
              weekflag(i)=0;
           end
       end
   end
end
weektrain = weekflag(1:numtrain);
weektest = weekflag(numtrain+1:numtrain+numtest);
% % 
traindata=traindata(weektrain==week,:);
trainlabels=trainlabels(weektrain==week,:);
testdata=testdata(weektest==week,:);
testlabels=testlabels(weektest==week,:);

% 分白天晚上
daytime=zeros(96,1);
daytime(21:84)=1; %早上6点到晚上8点
traindaytime=repmat(daytime,size(traindata,1)/96,1);
testdaytime=repmat(daytime,size(testdata,1)/96,1);
% 
traindata = traindata(traindaytime==day,:);
trainlabels = trainlabels(traindaytime==day,:);
testdata = testdata(testdaytime==day,:);
testlabels = testlabels(testdaytime==day,:);



% model=cell(1);
% model{1}.W=0.1*randn(size(traindata,2),hidelayer);
% model{1}.Wc=0.1*randn(hidelayer,size(trainlabels,2));
% model{1}.b=zeros(1,hidelayer);
% model{1}.cc=zeros(1,size(trainlabels,2));

%% 训练
addpath RBM/
numhide = size(hidelayer,2);
op1.verbose=true;
op1.maxepoch=50;
op2=op1;
op3=op1;
op4=op1;
op5=op1;
model=dbnFit(traindata,hidelayer,trainlabels,topfunc,hidefunc,op1,op2,op3,op4,op5); %训练
toc;

tic;
addpath dbn/
options.Method = 'scg';
options.display = 'off';
options.maxIter = 4000;
[m,cost] = bpFine(model, traindata, trainlabels, topfunc, hidefunc,options);
% [m] = bp( model, traindata, trainlabels, topfunc, hidefunc);
toc;


numtest = size(testdata,1);
a = runnet(testdata, m.W, m.b, numhide,topfunc,hidefunc);
h = a{numhide+2};
dp=mapminmax('reverse',h,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;
dp(dp<1)=5;
re=sum(abs(dp-dr)./dr)/numtest;
count=0;
l='';
for i=1:numlink
    if re(i)>1
%       re(i)=0;
        s=sprintf('%d ',i);
        l=[l s];
        count=count+1;
    end
end
MRE = sum(re)/(numlink-count); 
MAE = sum(sum(abs(dp-dr)))/(numlink*numtest);
RMSE = sqrt(sum(sum((dp-dr).^2))/(numlink*numtest));
t2=clock;
time=etime(t2,t1);
plot(1:numlink,re,'*r');
if isequal(topfunc,@Softplus)
    topfunc=@sp;
end
if isequal(hidefunc,@Softplus)
    hidefunc=@sp;
end
log=sprintf('%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t%s',func2str(topfunc),func2str(hidefunc),options.Method,options.maxIter,numperiod,numhide,hidelayer(1),week,day,cost,MRE,MAE,RMSE,time,l);
filename=sprintf('~/hg/testResult/DBN_pemsd05_stationNew147_train71_test18.txt');
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', log);
fclose(fp);