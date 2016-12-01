function [cost,grad]=rnn_cost(theta,input_size,hide_size,output_size,data,labels,topfunc,hidefunc,delay)

net=rnn_thetatonet(theta,input_size,hide_size,output_size);
[s,o] = rnn_forward(data,net,topfunc,hidefunc,delay); % !!!!
% [data,labels]=divide_data(data, labels, delay); %totalRNN
y=labels;
squared_error=0.5*sum((y-o).^2,1);
m=size(labels,1);
cost=1/m*sum(squared_error);

% 计算更新梯度
m=size(labels,1);

% V_delta = -(y-o).*funcdiff(topfunc,o);
V_delta = -(y-o);
V_grad = 1/m*(s{delay}'*V_delta);

e=cell(delay+1);
e{delay+1}=V_delta;
e{delay}= (e{delay+1}*net.V').*funcdiff(hidefunc,s{delay});
for i=delay-1:-1:1
    e{i} = (e{i+1}*net.W').*funcdiff(hidefunc,s{i});
end
% e4 = V_delta;
% e3 = (e4*net.V').*funcdiff(hidefunc,s{3});
% e2 = (e3*net.W').*funcdiff(hidefunc,s{2});
% e1 = (e2*net.W').*funcdiff(hidefunc,s{1});
sum_Ugrad=zeros(size(net.U));
for i=1:delay
    sum_Ugrad=sum_Ugrad+data{i}'*e{i};
end
U_grad = 1/m*sum_Ugrad;
% U_grad = 1/m*(data{3}'*e3+data{2}'*e2+data{1}'*e1);

sum_Wgrad=zeros(size(net.W));
for i=1:delay-1
    sum_Wgrad=sum_Wgrad+s{i}'*e{i+1};
end
W_grad = 1/m*(sum_Wgrad);
% W_grad = 1/m*(s{2}'*e3+s{1}'*e2);

sum_bgrad=zeros(size(net.b));
for i=1:delay
    sum_bgrad=sum_bgrad+e{i};
end
b_grad = 1/m*sum(sum_bgrad,1);
% b_grad = 1/m*sum(e3+e2+e1,1);

c_grad = 1/m*sum(V_delta,1);

grad=[U_grad(:);V_grad(:);W_grad(:);b_grad(:);c_grad(:)];

end