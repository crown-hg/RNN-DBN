function [cost,grad]=change_dbn_cost(theta,input_size,hide_size,output_size,data,labels,topfunc,hidefunc,delay)

net=change_dbn_thetatonet(theta,input_size,hide_size,output_size,delay);
[s,o] = change_dbn_forward(data,net,topfunc,hidefunc,delay);
y=labels;
m=size(labels,1);
% 均方误差
% squared_error=0.5*sum((y-o).^2,1);
% cost=1/m*sum(squared_error);
% 交叉熵误差
cost=-1/(m*147)*sum(sum(y.*log(o)+(1-y).*log(1-o)));
% 计算更新梯度
% V_delta = -(y-o).*funcdiff(topfunc,o); %均方代价
V_delta = -(y-o); %交叉熵代价

Vgrad = 1/m*(s{delay}'*V_delta);
cgrad = 1/m*sum(V_delta,1);

delta=cell(delay+1);
delta{delay+1}=V_delta;
delta{delay}= (delta{delay+1}*net{delay}.V').*funcdiff(hidefunc,s{delay});
for i=delay-1:-1:1
    delta{i} = (delta{i+1}*net{i+1}.W').*funcdiff(hidefunc,s{i}); % 看模型，delta2=（delta3*net3.W'）.*f'(s2)
end
% e4 = V_delta;
% e3 = (e4*net.V').*funcdiff(hidefunc,s{3});
% e2 = (e3*net.W').*funcdiff(hidefunc,s{2});
% e1 = (e2*net.W').*funcdiff(hidefunc,s{1});
Ugrad=cell(delay,1);
Wgrad=cell(delay,1);
bgrad=cell(delay,1);
Ugrad{1}=1/m*(data{1}'*delta{1});
bgrad{1}=1/m*sum(delta{1},1);
for i=2:delay
    Ugrad{i}=1/m*(data{i}'*delta{i});
    Wgrad{i}=1/m*(s{i-1}'*delta{i});
    bgrad{i}=1/m*sum(delta{i},1);
end
grad=[];
grad=[grad;Ugrad{1}(:);bgrad{1}(:)];
for i=2:delay
    grad=[grad;Ugrad{i}(:);Wgrad{i}(:);bgrad{i}(:)];
end
grad=[grad;Vgrad(:);cgrad(:)];
end