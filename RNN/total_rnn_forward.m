function [s,o]=total_rnn_forward(data,net,topfunc,hidefunc,delay)
% m样本数
m=size(data,1);
[hidesize,outsize]=size(net.V);
sh=zeros(m,hidesize);
o=zeros(m-delay+1,outsize);
sh(1,:)=hidefunc(data(1,:)*net.U+net.b);
for n=2:m
    sh(n,:)=hidefunc(data(n,:)*net.U+net.b+sh(n,:)*net.W);
    if n>2
        o(n-2,:)=topfunc(sh(n,:)*net.V+net.c);
    end
end
s=cell(1,3);
for i=1:delay
    s{i}=sh(i:m-delay+i,:);
end
end