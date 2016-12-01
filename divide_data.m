function [data] = divide_data( traindata, delay )
% 把训练数据分成delay份
% 一共delay个时刻的数据
numlink=size(traindata,2)/delay;
place=cell(1,delay);
data=cell(1,delay);
for i=1:delay
    place{i}=zeros(1,delay);
    place{i}(1,i)=1;
    place{i}=repmat(place{i},1,numlink);
    data{i}=traindata(:,place{i}==1);
end