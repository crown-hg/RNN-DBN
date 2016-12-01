function [model, errors] = rbm(X, numhid, hidefunc, varargin)
% 这个函数是用来建立一个两层的RBM，V的维度是X的列数，H的维度是numhid
% varargin是用来设置RBM的参数的，通常把verbose设成true，就可以了，其他默认就行，当然也可以根据需求来改
%Learn RBM with Bernoulli hidden and visible units
%This is not meant to be applied to image data
%code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] to be interpreted 训练数据
%               ... as probabilities
%numhid         ... number of hidden layers隐层节点数

%additional inputs (specified as name value pairs or in struct)
%method         ... CD or SML 
%eta            ... learning rate 动量学习率
%momentum       ... momentum for smoothness amd to prevent overfitting 动量，快速收敛防止过拟合
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data 最大训练次数
%avglast        ... how many epochs before maxepoch to start averaging
%               ... before. Procedure suggested for faster convergence by
%               在使用平均化之前要运行多少次
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor 权重衰减项
%batchsize      ... The number of training instances per batch 训练集中样本的个数
%verbose        ... For printing progress 这个东西就是为了打印那一行字
%anneal         ... Flag. If set true, the penalty is annealed linearly
%               ... through epochs to 10% of its original value

%OUTPUTS:
%model.type     ... Type of RBM (i.e. type of its visible and hidden units)
%model.W        ... The weights of the connections
%model.b        ... The biases of the hidden layer
%model.c        ... The biases of the visible layer
%model.top      ... The activity of the top layer, to be used when training
%               ... DBN's
%errors         ... The errors in reconstruction at every epoch

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   method        ...
    eta           ...
    momentum      ...
    maxepoch      ...
    avglast       ...
    penalty       ...
    batchsize     ...
    verbose       ...
    anneal        ...
    ] = process_options(args    , ...
    'method'        ,  'CD'     , ...
    'eta'           ,  0.1      , ...
    'momentum'      ,  0.5      , ...
    'maxepoch'      ,  20       , ...
    'avglast'       ,  5        , ...
    'penalty'       , 2e-4      , ...
    'batchsize'     , 100       , ...
    'verbose'       , false     , ...
    'anneal'        , false);
avgstart = maxepoch - avglast;
oldpenalty= penalty;
[N,d]=size(X);

if (verbose) 
    fprintf('Preprocessing data...\n');
end

%Create batches
numcases=N;
numdims=d;
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
perm=randperm(N);
groups = groups(perm);
for i=1:numbatches
    batchdata{i}= X(groups==i,:);
end

%train RBM 
W = 0.1*randn(numdims,numhid);
c = zeros(1,numdims);
b = zeros(1,numhid);
ph = zeros(numcases,numhid);
nh = zeros(numcases,numhid);
phstates = zeros(numcases,numhid);
nhstates = zeros(numcases,numhid);
negdata = zeros(numcases,numdims);
negdatastates = zeros(numcases,numdims);
Winc  = zeros(numdims,numhid); %权值更新的大小
binc = zeros(1,numhid); %隐层偏移更新的大小
cinc = zeros(1,numdims); %显层偏移更新的大小
Wavg = W;
bavg = b;
cavg = c;
t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    
	errsum=0;
    if (anneal)
        %apply linear weight penalty decay
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases, numdims]=size(batchdata{batch});
		data = batchdata{batch};
        
        %go up 第一次从显层到隐层
		ph = hidefunc(data*W + repmat(b,numcases,1));
		phstates = ph > rand(numcases,numhid);
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
		
        %go down 第一次从隐层到显层
		negdata = hidefunc(nhstates*W' + repmat(c,numcases,1));
		negdatastates = negdata > rand(numcases,numdims);

        %go up one more time第二次显层到隐层
		nh = hidefunc(negdatastates*W + repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
        %update weights and biases
        dW = (data'*ph - negdata'*nh);  %v1*h1-v2h2 W的更新，这里是伯努利分布的所以用的negdatastate，如果不是则可以改成negdata
        dc = sum(data) - sum(negdata);  %v1 - v2显层偏移的更新
        db = sum(ph) - sum(nh); %h1-h2 隐层偏移的更新
		Winc = momentum*Winc + eta*(dW/numcases - penalty*W); %W的带动量的更新的大小
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases); 
		W = W + Winc; %W的更新
		b = b + binc;
		c = c + cinc;
        
        if (epoch > avgstart)
            %超过45次apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);%其实这一步就是把W的改变量从Winc变成了1/t*Winc,也就是减小了改变量
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
        end
        
        %accumulate reconstruction error
        err= sum(sum( (data-negdata).^2 )); %这里为什么用negdata而不是negdatastates？
		errsum = err + errsum;
    end
    
    errors(epoch)=errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i. Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
    end
end

model.top= hidefunc(X*Wavg + repmat(bavg,N,1));
model.W= Wavg;
model.b= bavg;
model.c= cavg;
