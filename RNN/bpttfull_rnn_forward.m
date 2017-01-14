function [sh,o]=bpttfull_rnn_forward(data,net,topfunc,hidefunc)
% 这里是真正的RNN迭代过程，每一步都是从之前的步骤得到

m=size(data,1); % m样本数，data是numperiod为1的数据
[hidesize,outsize]=size(net.V); % 隐层和输出层
sh=zeros(m,hidesize); % 保存产生的所有隐层的数据
o=zeros(m, outsize); % 最终输出的数据
sh(1,:)=hidefunc(data(1,:)*net.U+net.b); % 第一个隐层
o(1,:)=topfunc(sh(1,:)*net.V+net.c);
for n=2:m
    sh(n,:)=hidefunc(data(n,:)*net.U+sh(n-1,:)*net.W+net.b);
    o(n,:)=topfunc(sh(n,:)*net.V+net.c); % 因为bptt依靠delay来进行更新的，所以要少
end
end