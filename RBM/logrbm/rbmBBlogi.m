function [model, errors] = rbmBBlogi(X, numhid, varargin)
% �����������������һ�������RBM��V��ά����X��������H��ά����numhid
% varargin����������RBM�Ĳ����ģ�ͨ����verbose���true���Ϳ����ˣ�����Ĭ�Ͼ��У���ȻҲ���Ը�����������
%Learn RBM with Bernoulli hidden and visible units
%This is not meant to be applied to image data
%code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] to be interpreted ѵ������
%               ... as probabilities
%numhid         ... number of hidden layers����ڵ���

%additional inputs (specified as name value pairs or in struct)
%method         ... CD or SML 
%eta            ... learning rate ����ѧϰ��
%momentum       ... momentum for smoothness amd to prevent overfitting ����������������ֹ�����
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data ���ѵ������
%avglast        ... how many epochs before maxepoch to start averaging
%               ... before. Procedure suggested for faster convergence by
%               ��ʹ��ƽ����֮ǰҪ���ж��ٴ�
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor Ȩ��˥����
%batchsize      ... The number of training instances per batch ѵ�����������ĸ���
%verbose        ... For printing progress �����������Ϊ�˴�ӡ��һ����
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
    'maxepoch'      ,  50       , ...
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
Winc  = zeros(numdims,numhid); %Ȩֵ���µĴ�С
binc = zeros(1,numhid); %����ƫ�Ƹ��µĴ�С
cinc = zeros(1,numdims); %�Բ�ƫ�Ƹ��µĴ�С
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
        
        %go up ��һ�δ��Բ㵽����
		ph = logistic(data*W + repmat(b,numcases,1));
		phstates = ph > rand(numcases,numhid);
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
		
        %go down ��һ�δ����㵽�Բ�
		negdata = logistic(nhstates*W' + repmat(c,numcases,1));
		negdatastates = negdata > rand(numcases,numdims);

        %go up one more time�ڶ����Բ㵽����
		nh = logistic(negdatastates*W + repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
        %update weights and biases
        dW = (data'*ph - negdata'*nh);  %v1*h1-v2h2 W�ĸ��£������ǲ�Ŭ���ֲ��������õ�negdatastate�������������Ըĳ�negdata
        dc = sum(data) - sum(negdata);  %v1 - v2�Բ�ƫ�Ƶĸ���
        db = sum(ph) - sum(nh); %h1-h2 ����ƫ�Ƶĸ���
		Winc = momentum*Winc + eta*(dW/numcases - penalty*W); %W�Ĵ������ĸ��µĴ�С
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases); 
		W = W + Winc; %W�ĸ���
		b = b + binc;
		c = c + cinc;
        
        if (epoch > avgstart)
            %����45��apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);%��ʵ��һ�����ǰ�W�ĸı�����Winc�����1/t*Winc,Ҳ���Ǽ�С�˸ı���
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
        end
        
        %accumulate reconstruction error
        err= sum(sum( (data-negdata).^2 )); %����Ϊʲô��negdata������negdatastates��
		errsum = err + errsum;
    end
    
    errors(epoch)=errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i. Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
    end
end

model.type= 'BB';
model.top= logistic(X*Wavg + repmat(bavg,N,1));
model.W= Wavg;
model.b= bavg;
model.c= cavg;
