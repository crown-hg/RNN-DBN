addpath data/
load('new147k4');
numtrain = 71*96; %前9个月，3月10日和9月17日是去掉的
numtest = 18*96; %第10月
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);
nh = 100;
out = 100;
epoch=1000;
tf1='logsig';
tf2='tansig';
net = newff([0 8],[10 1],{'tansig' 'purelin'},'trainlm'); 
net.trainParam.epochs=epoch;
net.trainParam.goal=0.01;
net.trainParam.min_grad=1e-20;
net.trainParam.show=20;
net.trainParam.time=inf;
net.trainFcn='trainblm';
net=train(net,traindata',trainlabels');