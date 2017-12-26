function [cost,grad]=rnn_cost(theta,input_size,hide_size,output_size,data,labels,testdata, testlabels, ps, topfunc,hidefunc,delay)

net=rnn_thetatonet(theta,input_size,hide_size,output_size);
[s,o] = rnn_forward(data,net,topfunc,hidefunc,delay); % !!!!
% [data,labels]=divide_data(data, labels, delay); %totalRNN
y=labels;
[m,nl]=size(labels);
ws=sum(theta.^2);
% 均方误差
lamda=0.00001;
squared_error=0.5*sum((y-o).^2,1);
cost=1/m*sum(squared_error)+lamda*ws;
% 交叉熵误差
% cost=-1/(m*nl)*sum(sum(y.*log(o)+(1-y).*log(1-o)));

global show logresult;
show=show+1;
if mod(show,10)==0
numlink=size(labels,2);
[~,out]=rnn_forward(testdata,net,topfunc,hidefunc,delay);
numtest = size(testlabels,1);
dp=mapminmax('reverse',out,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;
dp(dp<1)=1;
re=sum(abs(dp-dr)./dr)/numtest;
MRE = sum(re)/numlink;
MAE = sum(sum(abs(dp-dr)))/(numlink*numtest);
RMSE = sqrt(sum(sum((dp-dr).^2))/(numlink*numtest));

logresult(show,1)=MRE;
logresult(show,2)=MAE;
logresult(show,3)=RMSE;
fprintf('mre %.4f  mae %.2f  rmse %.2f ',MRE,MAE,RMSE);
end
% 计算更新梯度
V_delta = -(y-o).*funcdiff(topfunc,o); %均方
% V_delta = -(y-o); %交叉熵
V_grad = 1/m*(s{delay}'*V_delta)+lamda*net.V;

e=cell(delay+1,1);
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
U_grad = 1/m*sum_Ugrad+lamda*net.U;
% U_grad = 1/m*(data{3}'*e3+data{2}'*e2+data{1}'*e1);

sum_Wgrad=zeros(size(net.W));
for i=1:delay-1
    sum_Wgrad=sum_Wgrad+s{i}'*e{i+1};
end
W_grad = 1/m*(sum_Wgrad)+lamda*net.W;
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