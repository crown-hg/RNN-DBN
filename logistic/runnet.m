function [ a ] = runnet( data, W, b, numhide )
% 输入数据data，权值W，bias b
% 获取结果h
m=size(data,1);
%var=ones(m,1)*sigma;
a=cell(1,numhide+2);
a{1}=data;
%a{2}=logistic((data./var)*W{1} + repmat(b{1},m,1));
for i=2:numhide+2
    a{i}=logistic(a{i-1}*W{i-1}+repmat(b{i-1},m,1));
end
end