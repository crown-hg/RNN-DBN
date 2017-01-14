function [ m ] = bpauto( model, X,numhide,hidefunc )
%这里把搭建的多层rbm进行自编码，高层使用rbm中的Ｗ的转置来初始化
numW=numhide*2;
W=cell(numW,1);
b=cell(numW,1);
theta=[];
btheta=[];
for i=1:numhide
    W{i}=model{i}.W;
    theta=[theta; W{i}(:)];
    b{i}=model{i}.b;
    btheta=[btheta; b{i}(:)];
end
for i=numhide+1:numW
    W{i}=model{numW+1-i}.W';
    theta=[theta; W{i}(:)];
    b{i}=model{numW+1-i}.c;
    btheta=[btheta; b{i}(:)];
end
theta=[theta;btheta];

[vn,hn]=size(W{1});
addpath minFunc/
options.Method = 'scg';
options.display = 'on';
options.maxIter = 200;
[opttheta, ~] = minFunc( @(p) bpcost( p ,...
                    numW-1,vn ,vn,hn, 0.0001, X, X, hidefunc, hidefunc,'square'),...
                     theta, options);

[Wt, bt]=thetatowb(opttheta, numW-1, vn, vn, hn);
m=cell(numhide);
for i=1:numhide
    m{i}.W=Wt{i};
    m{i}.b=bt{i};
end
end