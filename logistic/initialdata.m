function [ data_s,md,sd ] = initialdata( data )
% 数据归一化
% data是n*T的
% n是维度，T是样本数

md = mean(data,1); % 样本均值
datat = bsxfun(@minus,data,md);
sd = std(datat,0,1); % 样本标准差
sd(sd==0)=1;
data_s = bsxfun(@rdivide,datat,sd);
end

