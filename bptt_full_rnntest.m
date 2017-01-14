% agruname = sprintf('topFunc\thFunc\tdelay\tNH\tbpfunc\tbpmax\tWEEK\tDAY\tCOST\tMRE\tMAE\tTIME\tBADLINK');
% filename=sprintf('~/hg/testResult/full_bptt_RNN.txt');
% fp = fopen(filename,'wt'); 
% fprintf(fp, '%s', agruname);
% fclose(fp);
t1=clock;
week = 10; %1是weekday工作日，0是weekend双休日
day = 10;%1为daytime， 0为nighttime

%% 取数据
delay=1;
addpath ../data/
dataname=sprintf('new147k1');
load(dataname);

numlink=size(labels,2);
numtrain = 71*96; %前9个月，3月10日和9月17日是去掉的
numtest = 18*96; %第10月
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);

% numlink=200;
% load('/home/hg/Code/data/train567test8k1.mat');
% traindata=train.data;
% trainlabels=train.labels;
% testdata=test.data;
% testlabels=test.labels;

%% 建模
addpath RNN/
opts.maxepoch=300;
opts.inputsize = numlink;
opts.hidesize = 100;
opts.outputsize = numlink;
opts.topfunc=@tanh;
opts.hidefunc=@tanh;
% 初始化参数
ru  = sqrt(6) / sqrt(opts.inputsize+opts.hidesize);
U = 0.5*(rand(opts.inputsize,opts.hidesize)*2*ru-ru);
rv  = sqrt(6) / sqrt(opts.inputsize+opts.hidesize);
V = rand(opts.hidesize,opts.outputsize)*2*rv-rv;
rw  = sqrt(6) / sqrt(opts.inputsize+opts.hidesize);
W = 0.5*(rand(opts.hidesize,opts.hidesize)*2*rw-rw);
b = zeros(1,opts.hidesize);
c = zeros(1,opts.outputsize);
theta=[U(:);V(:);W(:);b(:);c(:)];
% 训练
addpath minFunc/
minfunc_options.Method = 'scg';
minfunc_options.display = 'on';
minfunc_options.maxIter =3000;
[theta, cost] = minFunc( @bpttfull_rnn_cost ,theta, minfunc_options, traindata,trainlabels,opts);
net =rnn_thetatonet(theta,opts.inputsize,opts.hidesize,opts.outputsize);

%% 测试
% [sss,out]=rnn_forward(testdata,net,topfunc,hidefunc,delay);
% testdata=traindata;
% testlabels=trainlabels;
[s,out]=bpttfull_rnn_forward(testdata,net,opts.topfunc,opts.hidefunc);
numtest = size(testlabels,1);
dp=mapminmax('reverse',out,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1; 
dp(dp<=0)=3;
re=sum(abs(dp-dr)./dr)/numtest;
count=0;
l='';
for i=1:numlink
    if re(i)>0.5
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
plot(1:numlink,re,'r*');
result = sprintf('%s\t%s\t%d\t%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t%s',...
                func2str(opts.topfunc),func2str(opts.hidefunc),opts.hidesize,minfunc_options.Method,...
                minfunc_options.maxIter,week,day,cost,MRE,MAE,RMSE,time,l);
filename=sprintf('~/hg/testResult/full_bptt_RNN.txt');
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', result);
fclose(fp);