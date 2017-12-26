function [ model,cost ] = bpFine( m, data, labels, testdata, testlabels, ps, topfunc, hidefunc,options,costtype,lambda)
visibleSize=size(data,2);
hiddenSize = size(m{1}.W,2);
% hiddenSize=0;%##
numClasses = size(labels,2);
numhide = size(m,1);
% numhide=0;%##
% Wc=0.1*randn(hiddenSize,numClasses);% auto
% bc=zeros(1,numClasses);% auto
theta=[];
b=[];
for i=1:numhide
    theta = [theta;m{i}.W(:)];
    b=[b;m{i}.b(:)];
end
% theta = [theta;Wc(:)];% auto
% b=[b;bc(:)]; %auto 
WC=m{i}.Wc';
% WC=m{1}.Wc;%##
theta = [theta;WC(:)];
b = [b;m{i}.cc(:)];
% b = [b;m{1}.cc(:)];%##
theta = [theta;b];
addpath minFunc/
for i=1:1
[theta, cost] = minFunc( @(p) bpcost( p ,...
                    numhide,numClasses ,visibleSize,hiddenSize,lambda, data, labels, testdata, testlabels, ps, topfunc, hidefunc,costtype),...
                     theta, options);
end
[model.W, model.b]=thetatowb(theta, numhide, numClasses, visibleSize, hiddenSize);
end