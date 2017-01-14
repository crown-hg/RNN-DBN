function [sh,o]=bptt_rnn_forward(data,net,topfunc,hidefunc,delay)
% 这里是真正的RNN迭代过程，每一步都是从之前的步骤得到
m=size(data,1); % m样本数，data是numperiod为1的数据
[hidesize,outsize]=size(net.V); % 隐层和输出层
sh=zeros(m,hidesize); % 保存产生的所有隐层的数据
o=zeros(m, outsize); % 最终输出的数据
sh(1,:)=hidefunc(data(1,:)*net.U+net.b); % 第一个隐层
for n=2:delay-1
        sh(n,:)=hidefunc(data(n,:)*net.U+net.b+sh(n-1,:)*net.W);
end
for n=delay:m
    sh(n,:)=hidefunc(data(n,:)*net.U+net.b+sh(n-1,:)*net.W);
    o(n,:)=topfunc(sh(n,:)*net.V+net.c); % 因为bptt依靠delay来进行更新的，所以要少
end
end