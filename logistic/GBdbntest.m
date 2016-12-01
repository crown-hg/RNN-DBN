addpath data/
load('td-1_link200_day20.mat');
addpath RBM/
op.verbose=true;
numtrain = 1500;
numtest = 400;
numlink = 200;
hidelayer = [100 100 100];
numhide = size(hidelayer,2);
traindata1 = traindata(1:numtrain,:);
trainlabels1 = trainlabels(1:numtrain,:);
testdata = traindata(1:numtest,:);
testlabels = trainlabels(1:numtest,:);

batchsize=100;
N=numtrain;
d=size(traindata1,2);
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
perm=randperm(N);
groups = groups(perm);
batchdata=zeros(100,d,numbatches);
for i=1:numbatches
    batchdata(:,:,i)= traindata1(groups==i,:);
end

params.v_var=1;
params.epislonw_vng = 0.001;
params.std_rate=0.001;
params.maxepoch=100;
params.nHidNodes=100;
params.PreWts.vhW=0.1*randn(d,100);
params.PreWts.hb=zeros(1,100);
params.PreWts.vb=zeros(1,d);
params.nCD=1;
params.init_final_momen_iter=min(30,params.maxepoch/2);
params.init_momen=0.5;
params.final_momen=0.9;
params.wtcost=0.0002;
params.SPARSE=1;
params.sparse_p=0.01;
params.sparse_lambda=2;
[vhW, vb, hb, fvar, errs] = GRBM(traindata1, params);