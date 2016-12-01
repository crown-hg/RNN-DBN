function [net, errors] = rnn_rbmFit(net, s, X, numhid, y, topfunc, hidefunc, varargin)
%Fit an RBM to discrete labels in y
%This is not meant to be applied to image data
%code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as 训练数据
%               ... probabilities
%numhid         ... number of hidden units 隐层节点数
%y              ... List of discrete labels 标签

%additional inputs (specified as name value pairs or in struct) 
%nclasses       ... number of classes 类别个数
%method         ... CD or SML 
%eta            ... learning rate学习率
%momentum       ... momentum for smoothness amd to prevent overfitting 
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data 最大训练次数
%avglast        ... how many epochs before maxepoch to start averaging 
%               ... before. Procedure suggested for faster convergence by
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor权重衰减项
%weightdecay    ... A boolean flag. When set to true, the weights are
%               ... Decayed linearly from penalty->0.1*penalty in epochs
%batchsize      ... The number of training instances per batch训练集中训练数据的个数
%verbose        ... For printing progress打印过程
%anneal         ... Flag. If set true, the penalty is annealed linearly退火
%               ... through epochs to 10% of its original value

%OUTPUTS:
%model.W        ... The weights of the connections
%model.b        ... The biases of the hidden layer
%model.c        ... The biases of the visible layer
%model.Wc       ... The weights on labels layer
%model.cc       ... The biases on labels layer
%errors         ... The errors in reconstruction at every epoch

%Process options
args= prepareArgs(varargin);
[   nclasses      ...
    method        ...
    eta           ...
    momentum      ...
    maxepoch      ...
    avglast       ...
    penalty       ...
    batchsize     ...
    verbose       ...
    anneal        ...
    ] = process_options(args    , ...
    'nclasses'      , nunique(y), ... %标签的取值个数
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
[N,d]=size(X); %N=5000,d = 784

if (verbose)
    fprintf('Preprocessing data...\n')
end

targets=y;
u=size(y,2); %u是标签的维度

%Create batches
numbatches= ceil(N/batchsize); %向上取整 N=5000 batchsize=100 numbatches=50
groups= repmat(1:numbatches, 1, batchsize); % 1~50然后扩展到100个，就是5000个数
groups= groups(1:N); %这一步好像没变化？
groups = groups(randperm(N));% 随机打乱groups中的数据，其实就是把随机的下标中的值再赋值给自己
batchdata=cell(1,numbatches);
batchtargets=cell(1,numbatches);
batch_s=cell(1,numbatches);
for i=1:numbatches
    batchdata{i}= X(groups==i,:); %batchdata{i}就是对X随机取100行数据，一共取50次
    batchtargets{i}= targets(groups==i,:); %同样batchtargets对targets也取值，是与batchdata中数据相对应的标签
    batch_s{i}=s(groups==i,:); 
end

% fit RBM
numcases=N; %样本数目5000
numdims=d; %输入维度784
numclasses=u;

W = net.U; %numhid=100,这是输入参数
Wc = net.V';
c = zeros(1,numdims);
b = net.b;
cc = net.c;
Ww = net.W;
sb = zeros(1,numhid);

nhstates = zeros(numcases,numhid);
Winc  = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);
Wcinc = zeros(numclasses,numhid);
ccinc = zeros(1,numclasses);
Wwinc = zeros(numhid,numhid);
sbinc = zeros(1,numhid);

Wavg = W;                                     
bavg = b;
cavg = c;
Wcavg = Wc;
ccavg = cc;
Wwavg = Ww;
sbavg = sb;

t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    
	errsum=0;
    if (anneal)
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases, numdims]=size(batchdata{batch});
		data = batchdata{batch};
		classes = batchtargets{batch}; %Y的取值
        sd = batch_s{batch};
        %go up 向上
        sflag = any(s~=0);
        if sflag
            ph = hidefunc(sd*Ww+data*W + classes*Wc + repmat(b,numcases,1)); %第一次得到的隐层数据
        else
            ph = hidefunc(data*W + classes*Wc + repmat(b,numcases,1)); %第一次得到的隐层数据
        end
        %这个是同时训练两个RBM，第一个是从V层到隐层的数据，第二个是从顶层到隐层的数据
		phstates = ph > rand(numcases,numhid); %跟随机数比较把计算的数据变为0和1
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD')) 
            nhstates = phstates;
        end 
		
        %go down 向下
		negdata = hidefunc(nhstates*W' + repmat(c,numcases,1));%第一次向下
		negdatastates = negdata > rand(numcases,numdims);%向显层方向
        
		negclasses = topfunc(nhstates*Wc' + repmat(cc,numcases,1));
        negclasses=negclasses>rand(numcases,numclasses);
        
        negs = hidefunc(nhstates*Ww' + repmat(sb,numcases,1));%s方向
		negs_states = negs > rand(numcases,numhid);
        
        %go up one more time
        if sflag
            nh = hidefunc(negs_states*Ww+negdatastates*W + negclasses*Wc + ... 
            repmat(b,numcases,1));
        else
		nh = hidefunc(negdatastates*W + negclasses*Wc + ... 
            repmat(b,numcases,1)); %第二次显层顶层同时向隐层
        end
		nhstates = nh > rand(numcases,numhid);
        %update weights and biases
        dW = (data'*ph - negdatastates'*nh); %v1*h1-v2h2
        dc = sum(data) - sum(negdatastates); %v1 - v2
        db = sum(ph) - sum(nh); %h1-h2
        dWc = (classes'*ph - negclasses'*nh); %另一个v1*h1-v2*h2
        dcc = sum(classes) - sum(negclasses); %另一个v1-v2
        dWw = (sd'*ph - negs_states'*nh); 
        dsb = sum(sd) - sum(negs_states); 
        
		Winc = momentum*Winc + eta*(dW/numcases - penalty*W);
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases);
		Wcinc = momentum*Wcinc + eta*(dWc/numcases - penalty*Wc);
		ccinc = momentum*ccinc + eta*(dcc/numcases);
        Wwinc = momentum*Wwinc + eta*(dWw/numcases - penalty*Ww);
        sbinc = momentum*sbinc + eta*(dsb/numcases);
        
		W = W + Winc;
		b = b + binc;
		c = c + cinc;
		Wc = Wc + Wcinc;
		cc = cc + ccinc;
        Ww = Ww + Wwinc;
        sb = sb + sbinc;
		
        if (epoch > avgstart)
            %apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			Wcavg = Wcavg - (1/t)*(Wcavg - Wc);
			ccavg = ccavg - (1/t)*(ccavg - cc);
            Wwavg = Wwavg - (1/t)*(Wwavg - Ww);
            sbavg = sbavg - (1/t)*(sbavg - sb);
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
			Wcavg = Wc;
			ccavg = cc;
            Wwavg = Ww;
			sbavg = sb;
        end
        
        %accumulate reconstruction error
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
    end
    
    errors(epoch)= errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i, Reconsruction error is %f\n', ...
            epoch, maxepoch, errsum);
    end
end

net.U= Wavg;
net.b= bavg;
net.c= ccavg;
net.V= Wcavg';
net.W= Wwavg;

