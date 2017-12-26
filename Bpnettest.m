%% initial parameters
t1=clock;
daytimesize=96;
week = 0; % 1 is weekday,0 is weekend, 10 is all
day = 0;% 1 is daytime, 0 is nighttime, 10 is all
topfunc=@flinear;
hidefunc=@no;
lambda = 0.0005; % regularization

%% get PeMS data
numperiod=4;
addpath ../data/
dataname=sprintf('new147k%d',numperiod);
load(dataname);
startlink=0;
numlink=147;
data=data(:,4*startlink+1:4*startlink+4*numlink);
labels=labels(:,startlink+1:4*startlink+numlink);

numtrain = 71*96; 
numtest = 18*96;
traindata = data(1:numtrain,:); 
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);

% %% 閫氱敤
% % divide data into daytime and night 
% weekflag=ones(size(data,1),1);
% for i=1:size(data,1)
%    if i<=68*96
%        if mod(ceil(i/96)+1,7)==6||mod(ceil(i/96)+1,7)==0
%           weekflag(i)=0;
%        end
%    else
%        if i>68*96&&i<=258*96
%            if mod(ceil(i/96)+2,7)==6||mod(ceil(i/96)+2,7)==0
%               weekflag(i)=0;
%            end
%        else
%            if mod(ceil(i/96)+3,7)==6||mod(ceil(i/96)+3,7)==0
%               weekflag(i)=0;
%            end
%        end
%    end
% end
% weektrain = weekflag(1:numtrain);
% weektest = weekflag(numtrain+1:numtrain+numtest);

% testdata11=testlabels(weektest==1,:);
% testdata00=testlabels(weektest==0,:);
% testlabels11=testlabels(weektest==1,:);
% testlabels00=testlabels(weektest==0,:);

% 分白天晚上
% daytime=zeros(96,1);
% daytime(21:84)=1; %早上6点到晚上8点
% traindaytime=repmat(daytime,size(traindata,1)/96,1);
% testdaytime=repmat(daytime,size(testdata,1)/96,1);
% 
% testdata11 = testdata(testdaytime==1,:);
% testdata00 = testdata(testdaytime==0,:);
% 
% testlabels11 = testlabels(testdaytime==1,:);
% testlabels00 = testlabels(testdaytime==0,:);

% get English freeway data 
% load('/home/hg/Code/data/train567test8.mat');
% traindata=train.data;
% trainlabels=train.labels;
% testdata=test.data;
% testlabels=test.labels;

%% initial weights and bias
inputsize=size(traindata,2);
outsize=size(trainlabels,2);
hiddenSize = 0;
numhide=0;

r = sqrt(6) / sqrt(inputsize+outsize);
model.W{1}=rand(inputsize,outsize)*2*r-r;
model.b{1}=zeros(1,outsize);
% r = sqrt(6) / sqrt(hiddenSize+outsize);
% model.W{2}=rand(hiddenSize,outsize)*2*r-r;
% model.b{2}=zeros(1,outsize);

addpath dbn/
costtype='square'; % cross & square
visibleSize=size(traindata,2);
numClasses = size(trainlabels,2);
theta = [];
for i=1:numhide+1
    theta=[theta;model.W{i}(:)];
end
for i=1:numhide+1
    theta=[theta;model.b{i}(:)];
end
%% train
addpath minFunc/
for i=1:1
options.Method = 'scg';
options.display = 'on';
options.maxIter = 3000; 
[theta, cost] = minFunc( @(p) bpcost( p ,...
                    numhide,numClasses ,visibleSize,hiddenSize,lambda, traindata, ...
                    trainlabels, testdata, testlabels, ps, topfunc, hidefunc,costtype),theta, options);
end
[model.W, model.b]=thetatowb(theta, numhide, numClasses, visibleSize, hiddenSize);
% % [model, error] = sgd_bp( model, numhide, traindata, trainlabels, testdata, testlabels, ps, topfunc, hidefunc);

%% test
numtest = size(testdata,1);
a11 = runnet(testdata11, model.W, model.b, numhide,topfunc,hidefunc);
h11 = a11{numhide+2};
load WB.mat
topfunc=@logsig;
hidefunc=@logsig;
numhide=1;
a00 = runnet(testdata00, W, b, 1,topfunc,hidefunc);
h00 = a00{numhide+2};

h=[h11;h00];
testlabels=[testlabels11;testlabels00];
dp=mapminmax('reverse',h,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;
dp(dp<1)=1;
re=sum(abs(dp-dr)./dr)/numtest;
count=0;
l='';
for i=1:numlink
    if re(i)>1
%       re(i)=0;
        s=sprintf('%d ',i);
        l=[l s];
%         count=count+1;
    end
end
MRE = sum(re)/(numlink-count); 
MAE = sum(sum(abs(dp-dr)))/(numlink*numtest);
RMSE = sqrt(sum(sum((dp-dr).^2))/(numlink*numtest));
t2=clock;
time=etime(t2,t1);
plot(1:numlink,re,'*r');

%% record the result
if isequal(topfunc,@Softplus)
    topfunc=@sp;
end
if isequal(hidefunc,@Softplus)
    hidefunc=@sp;
end
log=sprintf('%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f',...
    datestr(date),func2str(topfunc),func2str(hidefunc),options.Method,options.maxIter,numperiod,week,day,cost,MRE,MAE,RMSE,time);
filename=sprintf('~/testResult/Onelayer_regression.txt');
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', log);
fclose(fp);