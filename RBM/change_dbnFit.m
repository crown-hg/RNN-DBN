function net= change_dbnFit(data, numhid, labels, delay, topfunc, hidefunc,varargin)
%fit a DBN to bianry data in X
% 建立一个DBN，层数是numhid的个数，每一个数就是一层的维度
%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... list of numbers of hidden units
%y              ... List of discrete labels

%OUTPUTS:
%model          ... A cell array containing models from all RBM's
%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400
net=cell(delay,1);
[numsamples, input_size]=size(data{1});
output_size=size(labels,2);
net_temp.U = 0.1*randn(input_size,numhid);
net_temp.V = 0.1*randn(numhid,output_size);
net_temp.W = 0.1*randn(numhid,numhid);
net_temp.b=zeros(1,numhid);
net_temp.c=zeros(1,output_size);
s=zeros(numsamples,numhid);

if(delay==1) %delay只有1的时候
    net{1} = rnn_rbmFit(net_temp, s, data{1}, numhid, labels, topfunc, hidefunc, varargin{1});
else
    [net{1},s] = rnn_rbm(net_temp, data{1}, s, numhid, hidefunc, varargin{1});
    for i=2:delay-1 %这里是获得除了第一层和top层之外的所有隐层
        [net{i},s] = rnn_rbm(net_temp, data{i}, s, numhid, hidefunc, varargin{1});
    end
   %这里是获得top层的RBM，另外标签是回归层，在top层之上
    net{delay} = rnn_rbmFit(net_temp, s, data{delay}, numhid, labels, topfunc, hidefunc, varargin{1});
end