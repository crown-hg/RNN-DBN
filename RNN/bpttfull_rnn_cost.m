function [ cost, grad ] = bpttfull_rnn_cost( theta,data,labels,opts )
%函数先把数据全部运行一遍，然后反向计算误差
net=rnn_thetatonet(theta,opts.inputsize,opts.hidesize,opts.outputsize);
[s,out] = bpttfull_rnn_forward(data,net,opts.topfunc,opts.hidefunc); 

y=labels;
[m,nl]=size(labels);
ws=sum(theta.^2);
% 均方误差
lambda=0.0001;
squared_error=0.5*sum((out-y).^2,1);
cost=1/m*sum(squared_error)+lambda/2*ws;
% 交叉熵误差
% cost=-1/(m*nl)*sum(sum(y.*log(o)+(1-y).*log(1-o)));
% 最后一次不能求W的梯度，所以单算

V_delta = -(y(m,:)-out(m,:)).*funcdiff(opts.topfunc,out(m,:)); %均方
Vgrad = s(m,:)'*V_delta;
Udelta= (V_delta*net.V').*funcdiff(opts.hidefunc,s(m,:));
Ugrad = data(m,:)'*Udelta;
bgrad = Udelta;
cgrad = V_delta;
pre_Udelta=Udelta;
sumUgrad=Ugrad;
sumVgrad=Vgrad;
sumWgrad=zeros(size(net.W));
sumbgrad=bgrad;
sumcgrad=cgrad;
for time = m-1:-1:1
    % 计算更新梯度
    % V_delta = -(y-o); %交叉熵
    V_delta = -(y(time,:)-out(time,:)).*funcdiff(opts.topfunc,out(time,:)); %均方
    Vgrad = s(time,:)'*V_delta;
    Udelta= (V_delta*net.V').*funcdiff(opts.hidefunc,s(time,:));
    Ugrad = data(time,:)'*Udelta;
    Wgrad = s(time,:)'*pre_Udelta;
    bgrad = Udelta;
    cgrad = V_delta;
    pre_Udelta = Udelta;
    
    sumUgrad=sumUgrad+Ugrad;
    sumVgrad=sumVgrad+Vgrad;
    sumWgrad=sumWgrad+Wgrad;
    sumbgrad=sumbgrad+bgrad;
    sumcgrad=sumcgrad+cgrad;
end
U_grad=1/m*sumUgrad+lambda*net.U;
V_grad=1/m*sumVgrad+lambda*net.V;
W_grad=1/m*sumWgrad+lambda*net.W;
b_grad=1/m*sumbgrad;
c_grad=1/m*sumcgrad;

grad=[U_grad(:);V_grad(:);W_grad(:);b_grad(:);c_grad(:)];
end

