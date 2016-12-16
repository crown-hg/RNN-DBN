function [ model,cost ] = bpFine( m, data, labels, topfunc, hidefunc,options,costtype)
lambda = 0.0001;
visibleSize=size(data,2);
% hiddenSize = size(m{1}.W,2);
hiddenSize=0;%##
numClasses = size(labels,2);
% numhide = size(m,1);
numhide=0;%##
theta=[];
b=[];

for i=1:numhide
    theta = [theta;m{i}.W(:)];
    b=[b;m{i}.b(:)];
end
WC=m{1}.Wc';%##
theta = [theta;WC(:)];
b = [b;m{1}.cc(:)];%##
theta = [theta;b];
addpath minFunc/
[opttheta, cost] = minFunc( @(p) bpcost( p ,...
                    numhide,numClasses ,visibleSize,hiddenSize,lambda, data, labels, topfunc, hidefunc,costtype),...
                     theta, options);

[model.W, model.b]=thetatowb(opttheta, numhide, numClasses, visibleSize, hiddenSize);
end