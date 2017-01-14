function model= dbnFit(X, hidelayer, y, topfunc, hidefunc,varargin)
%fit a DBN to bianry data in X
% 建立一个DBN，层数是numhid的个数，每一个数就是一层的维度
%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%hidelayer         ... list of numbers of hidden units
%y              ... List of discrete labels

%OUTPUTS:
%model          ... A cell array containing models from all RBM's

%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400

numopts=length(varargin); %op的个数
H=length(hidelayer); %层数
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
    model{H}= rbmFit(X, hidelayer(end), y, topfunc, hidefunc, varargin{H});
else
    if(numopts>=1) %这里是获得visible层的RBM，如果等于0，我们对RBM的参数全取默认值
        model{1}= rbm(X, hidelayer(1), hidefunc,varargin{1});
    else
        model{1}= rbm(X, hidelayer(1), hidefunc);
    end
    %train all other RBM's on top of each other
    for i=2:H-1 %这里是获得除了visible层和top层之外的所有隐层
        if(numopts>=i) %小于i，则表明之后的模型全取默认值
            model{i}=rbm(model{i-1}.top, hidelayer(i), hidefunc, varargin{i});
        else
            model{i}=rbm(model{i-1}.top, hidelayer(i), hidefunc);
        end
    end
%     model=bpauto(model, X,H-1,hidefunc);
% the last RBM has access to labels too
    if(numopts>=H)%这里是获得top层的RBM，另外标签是回归层，在top层之上
        model{H}= rbmFit(model{H-1}.top, hidelayer(end), y, topfunc, hidefunc, varargin{H});
    else
        model{H}= rbmFit(model{H-1}.top, hidelayer(end), y, topfunc, hidefunc);
    end
end