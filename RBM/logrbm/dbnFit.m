function model= dbnFit(X, numhid, y, rbmFit, rbm,varargin)
%fit a DBN to bianry data in X
% ����һ��DBN��������numhid�ĸ�����ÿһ��������һ���ά��
%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... list of numbers of hidden units
%y              ... List of discrete labels

%OUTPUTS:
%model          ... A cell array containing models from all RBM's

%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400

numopts=length(varargin); %op�ĸ���
H=length(numhid); %����
model=cell(H,1);
%train the first RBM on data
% d=size(X,2);
% params.v_var=1;
% params.epislonw_vng = 0.001;
% params.std_rate=0.001;
% params.maxepoch=50;
% params.nHidNodes=numhid(1);
% params.PreWts.vhW=0.1*randn(d,numhid(1));
% params.PreWts.hb=zeros(1,numhid(1));
% params.PreWts.vb=zeros(1,d);
% params.nCD=1;
% params.init_final_momen_iter=min(30,params.maxepoch/2);
% params.init_momen=0.5;
% params.final_momen=0.9;
% params.wtcost=0.0002;
% params.SPARSE=1;
% params.sparse_p=0.01;
% params.sparse_lambda=2;
% model{1}= GaussianRBM(X, params);
if(H==1)
    model{H}= rbmFit(X, numhid(end), y, varargin{H});
else
    if(numopts>=1) %�����ǻ��visible���RBM���������0�����Ƕ�RBM�Ĳ���ȫȡĬ��ֵ
        model{1}= rbm(X, numhid(1),varargin{1});
    else
        model{1}= rbm(X, numhid(1));
    end
    %train all other RBM's on top of each other
    for i=2:H-1 %�����ǻ�ó���visible���top��֮�����������
        if(numopts>=i) %С��i�������֮���ģ��ȫȡĬ��ֵ
            model{i}=rbm(model{i-1}.top, numhid(i), varargin{i});
        else
            model{i}=rbm(model{i-1}.top, numhid(i));
        end
    end

%the last RBM has access to labels too
    if(numopts>=H)%�����ǻ��top���RBM�������ǩ�ǻع�㣬��top��֮��
        model{H}= rbmFit(model{H-1}.top, numhid(end), y, varargin{H});
    else
        model{H}= rbmFit(model{H-1}.top, numhid(end), y);
    end
end