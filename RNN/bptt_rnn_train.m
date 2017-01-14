function [mre,mae,rmse,best_mre,best_mae,best_rmse,cost] = bptt_rnn_train(options, traindata,trainlabels,testdata,testlabels)
% 这里是要搭建一个rnn

hidelayer=options.hidelayer;
topfunc=options.topfunc;
hidefunc=options.hidefunc;
delay=options.delay;
ps=options.ps;
maxepoch = options.maxepoch;
input_size = size(traindata,2);
hide_size = hidelayer;
[numsamples, output_size] = size(trainlabels);
ru  = sqrt(6) / sqrt(input_size+hide_size);
U = rand(input_size,hide_size)*2*ru-ru;
rv  = sqrt(6) / sqrt(input_size+hide_size);
V = rand(hide_size,output_size)*2*rv-rv;
rw  = sqrt(6) / sqrt(input_size+hide_size);
W = rand(hide_size,hide_size)*2*rw-rw;
b = zeros(1,hide_size);
c = zeros(1,output_size);
mre=zeros(1,maxepoch);
mae=zeros(1,maxepoch);
rmse=zeros(1,maxepoch);
best_mre.mre=1;
best_mae.mae=1000;
best_rmse.rmse=1000;
lambda=0.0001;
momentum = 0.4;
alpha=0.01;
eta=1;
s=zeros(numsamples,hide_size); % 保存产生的所有隐层的数据
for epoch=1:maxepoch
    tic;
    costsum=0;
    s(1,:)=hidefunc(traindata(1,:)*U+b); % 第一个隐层
    for n=2:delay-1
        s(n,:)=hidefunc(traindata(n,:)*U+b+s(n-1,:)*W);
    end
    old_U_grad=zeros(size(U));
    old_V_grad=zeros(size(V));
    old_W_grad=zeros(size(W));
    old_b_grad=zeros(size(b));
    old_c_grad=zeros(size(c));
    for n=delay:numsamples
        s(n,:)=hidefunc(traindata(n,:)*U+b+s(n-1,:)*W);
        h=topfunc(s(n,:)*V+c); % 因为bptt依靠delay来进行更新的，所以要少

        y=trainlabels(n,:);
        
        sum_UVW=sum(U(:).^2)+sum(V(:).^2)+sum(W(:).^2);
        costone=0.5*sum((y-h).^2)+lambda/2*sum_UVW;
%         costone=0.5*sum((y-h).^2);
        costsum=costsum+costone;
        if isnan(costsum)
            aaa=1;
        end
        V_delta = -(y-h).*funcdiff(topfunc,h); %均方
%         V_delta = -(y-h); %交叉熵
        dv = s(n,:)'*V_delta+lambda*V;
%         V_grad = s(n,:)'*V_delta;

        e=cell(delay+1,1);
        e{delay+1}=V_delta;
        e{delay}= (e{delay+1}*V').*funcdiff(hidefunc,s(n,:));
        for i=delay-1:-1:1
            e{i} = (e{i+1}*W').*funcdiff(hidefunc,s(n-delay+i,:));
        end
        sum_Ugrad=zeros(size(U));
        for i=1:delay
            sum_Ugrad=sum_Ugrad+traindata(n-delay+i,:)'*e{i};
        end
        du = sum_Ugrad+lambda*U;
%         U_grad = sum_Ugrad;

        sum_Wgrad=zeros(size(W));
        for i=1:delay-1
            sum_Wgrad=sum_Wgrad+s(n-delay+i,:)'*e{i+1};
        end
        dw = sum_Wgrad+lambda*W;
%         W_grad = sum_Wgrad;

        sum_bgrad=zeros(size(b));
        for i=1:delay
            sum_bgrad=sum_bgrad+e{i};
        end
        db = sum(sum_bgrad,1);

        dc = sum(V_delta,1);
        
        Ugrad = momentum*old_U_grad + alpha*du;
        Vgrad = momentum*old_V_grad + alpha*dv;
        Wgrad = momentum*old_W_grad + alpha*dw;
        bgrad = momentum*old_b_grad + alpha*db;
        cgrad = momentum*old_c_grad + alpha*dc;
       
        U = U-eta*Ugrad;
        V = V-eta*Vgrad;
        W = W-eta*Wgrad;
        b = b-eta*bgrad;
        c = c-eta*cgrad;
        
        old_U_grad=Ugrad;
        old_V_grad=Vgrad;
        old_W_grad=Wgrad;
        old_b_grad=bgrad;
        old_c_grad=cgrad;
    end
%     if epoch < 10
%         eta=eta-epoch/10;
%     end
    cost=1/numsamples*costsum;
    net.U=U;
    net.V=V;
    net.W=W;
    net.b=b;
    net.c=c;
    [s,out]=total_rnn_forward(testdata,net,topfunc,hidefunc,delay);
    numtest = size(testlabels,1)-delay+1;
    dp=mapminmax('reverse',out(delay:end,:),ps);
    dr=mapminmax('reverse',testlabels(delay:end,:),ps);
    dr(dr==0)=1; 
    dp(dp<=0)=3;
    re=sum(abs(dp-dr)./dr)/numtest;
    mre(epoch) = sum(re)/output_size;
    mae(epoch) = sum(sum(abs(dp-dr)))/(output_size*numtest);
    rmse(epoch) = sqrt(sum(sum((dp-dr).^2))/(output_size*numtest));
    if mre(epoch)<best_mre.mre
        best_mre.mre=mre(epoch);
        best_mre.net=net;
    end
    if mae(epoch)<best_mae.mae
        best_mae.mae=mae(epoch);
        best_mae.net=net;
    end
    if rmse(epoch)<best_rmse.rmse
        best_rmse.rmse=rmse(epoch);
        best_rmse.net=net;
    end
    fprintf('epoch  %d  cost  %f  %.4f  %.2f  %.2f  %.2f\n',epoch,cost,mre(epoch),mae(epoch),rmse(epoch),toc);
end
end