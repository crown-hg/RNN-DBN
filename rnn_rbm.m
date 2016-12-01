% agruname = sprintf('topFunc\thFunc\tdelay\tNH\tbpfunc\tbpmax\tWEEK\tDAY\tCOST\tMRE\tMAE\tTIME\tBADLINK');
% fp = fopen('~/hg/testResult/RNN_RBM_pemsd05_stationNew147_train71_test18.txt','wt'); 
% fprintf(fp, '%s', agruname);
% fclose(fp);
t1=clock;
daytimesize=96;
week = 1; %1是weekday工作日，0是weekend双休日
day = 1;%1为daytime， 0为nighttime
hidelayer = 100;
topfunc=@tanh;
hidefunc=@logsig;

%% 取数据
delay=3;
addpath /home/crown/hg/实验代码/data/
dataname=sprintf('new147k%d',delay);
load(dataname);

numlink=size(labels,2);
numtrain = 71*96; %前9个月，3月10日和9月17日是去掉的
numtest = 18*96; %第10月
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);

traindata = divide_data( traindata, delay );
testdata = divide_data( testdata, delay );

%% 分week和weekend
% 分week
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
trainnumsamples=size(trainlabels,1);
wtrain=weektrain(delay+1:trainnumsamples);

for i=1:delay
traindata{i}=traindata{i}(wtrain==week,:);
end
trainlabels=trainlabels(wtrain==week,:);
testnumsamples=size(testlabels,1);
wtest=weektest(delay+1:testnumsamples);
for i=1:delay
testdata{i}=testdata{i}(wtest==week,:);
end
testlabels=testlabels(wtest==week,:);

% 分day
daytime=zeros(96,1);
daytime(21:84)=1; %早上6点到晚上8点
traindaytime=repmat(daytime,52,1);
testdaytime=repmat(daytime,12,1);
traindaytime=traindaytime(delay+1:52*96);
testdaytime=testdaytime(delay+1:12*96);

for i=1:delay
traindata{i}=traindata{i}(traindaytime==day,:);
end
trainlabels=trainlabels(traindaytime==day,:);
for i=1:delay
testdata{i}=testdata{i}(testdaytime==day,:);
end
testlabels=testlabels(testdaytime==day,:);

%% 建模
addpath RBM/
op1.verbose=true;
op1.maxepoch=50;
net=rnn_dbnFit(traindata,hidelayer,trainlabels,delay,topfunc,hidefunc,op1,op1); %训练

addpath RNN/
options.Method = 'scg';
options.display = 'on';
options.maxIter =3000;
[net,cost]=rnn_train(options,net,traindata,trainlabels,hidelayer,topfunc,hidefunc,delay);

%% 测试
% [sss,out]=rnn_forward(testdata,net,topfunc,hidefunc,delay);
[s,out]=rnn_forward(testdata,net,topfunc,hidefunc,delay);
numtest = size(testlabels,1);
dp=mapminmax('reverse',out,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
dp(dp<1)=5;
re=sum(abs(dp-dr)./dr)/numtest;
count=0;
l='';
for i=1:numlink
    if re(i)>1
      re(i)=0;
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
plot(1:numlink,re,'r.');
if isequal(topfunc,@Softplus)
    topfunc=@sp;
end
if isequal(hidefunc,@Softplus)
    hidefunc=@sp;
end
result = sprintf('%s\t%s\t%d\t%d\t%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t%s',...
                func2str(topfunc),func2str(hidefunc),delay,hidelayer(1),options.Method,...
                options.maxIter,week,day,cost,MRE,MAE,RMSE,time,l);
filename=sprintf('~/hg/testResult/RNN_RBM_pemsd05_stationNew147_train71_test18.txt');
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', result);
fclose(fp);