function [net,cost] = change_dbn_train(options,net,data,labels,hidelayer,topfunc,hidefunc,delay)
% 这里是要搭建一个rnn

input_size = size(data{1},2);
hide_size = hidelayer;
[~,output_size] = size(labels);
theta=[];
theta=[theta;net{1}.U(:);net{1}.b(:)];
for i=2:delay
    theta=[theta;net{i}.U(:);net{i}.W(:);net{i}.b(:)];
end
theta = [theta;net{delay}.V(:);net{delay}.c(:)];
addpath minFunc/
[theta, cost] = minFunc( @(p) change_dbn_cost(p,input_size,hide_size,...
    output_size,data,labels,topfunc,hidefunc,delay),theta, options);

net =change_dbn_thetatonet(theta,input_size,hide_size,output_size,delay);
end