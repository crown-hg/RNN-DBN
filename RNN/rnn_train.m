function [net,cost] = rnn_train(options,net,data,labels,hidelayer,topfunc,hidefunc,delay)
% 这里是要搭建一个rnn

input_size = size(data{1},2);
hide_size = hidelayer;
[numsamples,output_size] = size(labels);
if isequal(net,1)
    U = 0.1*randn(input_size,hide_size);
    V = 0.1*randn(hide_size,output_size);
    W = 0.1*randn(hide_size,hide_size);
    b=zeros(1,hide_size);
    c=zeros(1,output_size);
else
    U = net.U;
    V = net.V;
    W = net.W;
    b = net.b;
    c = net.c;
end

theta=[U(:);V(:);W(:);b(:);c(:)];

addpath minFunc/
[theta, cost] = minFunc( @(p) rnn_cost(p,input_size,hide_size,...
    output_size,data,labels,topfunc,hidefunc,delay),theta, options);


% maxepoch = 100;
%  datatrain=cell(1,delay);
% for n=1:maxepoch
%     for j=1:numsamples
%         datatrain{1}=data{1}(j,:);
%         datatrain{2}=data{2}(j,:);
%         datatrain{3}=data{3}(j,:);
%         labelstrain=labels(j,:);
%         [cost,grad]=rnn_cost(theta,...
%         input_size,hide_size,output_size,datatrain,labelstrain,topfunc,hidefunc,delay);
%         theta =theta - 0.1*grad;
% %         fprintf('sample\t%d\tcost\t%f\n',j,cost);
%     end
%     net=rnn_thetatonet(theta,input_size,hide_size,output_size);
%     [s,o] = rnn_forward(data,net,topfunc,hidefunc,delay);
%     y=labels;
%     squared_error=0.5*sum((y-o).^2,1);
%     m=size(labels,1);
%     cost=1/m*sum(squared_error);
%     fprintf('epoch\t%d\tcost\t%f\n',n,cost);
% end

net =rnn_thetatonet(theta,input_size,hide_size,output_size);
end