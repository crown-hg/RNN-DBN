function [s,o]=rnn_forward(data,net,topfunc,hidefunc,delay)
% m样本数
m=size(data{1},1);
s=cell(1,delay);
for i=1:delay
    if i==1
        s{i}=hidefunc(data{i}*net.U+repmat(net.b,m,1));
    else
        s{i}=hidefunc(data{i}*net.U+repmat(net.b,m,1)+s{i-1}*net.W);
    end
end
o=topfunc(s{delay}*net.V+repmat(net.c,m,1));
end