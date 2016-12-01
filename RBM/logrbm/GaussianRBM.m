function [model, errs] = GaussianRBM(X, params)

batchsize=100;
[N,d]=size(X);
nBatches= ceil(N/batchsize);
groups= repmat(1:nBatches, 1, batchsize);
groups= groups(1:N);
perm=randperm(N);
groups = groups(perm);
batchdata=zeros(100,d,nBatches);
n=batchsize;
for i=1:nBatches
    batchdata(:,:,i)= X(groups==i,:);
end

assert(params.v_var > 0);
fstd = ones(1,d)*sqrt(params.v_var); %��׼����� params.v_var�Ǿ�����һ�������Ҿ�����һ����
params.v_var=[];

r = params.epislonw_vng; %r��ѧϰ��
 
std_rate = linspace(0, params.std_rate, params.maxepoch);
std_rate(:) = params.std_rate;
std_rate(1:min(30, params.maxepoch/2)) = 0; %learning schedule for variances����ѧϰ�ʵļƻ�������ÿ�η�����µĴ�С
 
assert( all(size(params.PreWts.vhW) == [d params.nHidNodes]));
assert( all(size(params.PreWts.hb) == [1 params.nHidNodes]));
assert( all(size(params.PreWts.vb) == [1 d]));
 
vhW = params.PreWts.vhW;
vb = params.PreWts.vb;
hb = params.PreWts.hb;
 
vhWInc  = zeros( d, params.nHidNodes);
hbInc   = zeros( 1, params.nHidNodes);
vbInc   = zeros( 1, d);
invfstdInc = zeros(1,d);

Ones = ones(n,1);

q=zeros(1, params.nHidNodes); %keep track of average activations ��¼ÿһ������ļ���ֵ
errs =  zeros(1, params.maxepoch); %���Ӧ���Ǽ�¼ÿһ�ε����
 
fprintf('\rTraining Learning v_var Gaussian-Binary RBM %d-%d   epochs:%d r:%f',...
    d, params.nHidNodes, params.maxepoch, r);
for epoch = 1:params.maxepoch
   
    %if rem(epoch, int32(params.maxepoch/20)) == 0 || epoch < 30
        fprintf('\repoch %d',epoch);
    %end
     
    errsum=0;
    ptot = 0;
    for batch = 1:nBatches
         
        Fstd = Ones*fstd;
         
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch); %nxd     
        pos_hidprobs = 1./(1 + exp(-(data./Fstd)*vhW - Ones*hb)); %p(h_j =1|data)��һ�λ��h��h1       
        pos_hidstates = pos_hidprobs > rand( size(pos_hidprobs) ); %h1ת��1��0
                 
        pos_prods    = (data./Fstd)'* pos_hidprobs; %1/sigma*v1*h1 �õ���û��0/1���������
        pos_hid_act  = sum(pos_hidprobs); %h1���ܼ���ֵ���õ�Ҳ��û��0/1���������
        pos_vis_act  = sum(data)./(fstd.^2); %see notes on this derivation v1/sima^2 ��������v���ƫ��
                
        %%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%
        for iterCD = 1:params.nCD % ncdָ���ǲ����Ĳ�����һ��Ϊ1
             
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdataprobs = pos_hidstates*vhW'.*Fstd+Ones*vb; %v2 �����õ���0/1��֮�������
            negdata = negdataprobs + randn(n, d).*Fstd; %�Ҿ����Ǽ�������
            neg_hidprobs = 1./(1 + exp(-(negdata./Fstd)*vhW - Ones*hb ));     %h2 updating hidden nodes again
            pos_hidstates = neg_hidprobs > rand( size(neg_hidprobs) );
             
        end %end CD iterations
        
        neg_prods  = (negdata./Fstd)'*neg_hidprobs; %1/sigma*v2*h2
        neg_hid_act = sum(neg_hidprobs); %h2�ļ���ֵ�ĺ�
        neg_vis_act = sum(negdata)./(fstd.^2); %see notes for detail  v2/sima^2
         
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
                
        errsum = errsum + sum(sum( (data-negdata).^2 )); %v1��v2�Ĳ�ֵ���ܺ�
         
        if epoch > params.init_final_momen_iter, %��ͬ��ѵ��ʱ�ڣ��в�ͬ�Ķ���
            momentum=params.final_momen;
        else
            momentum=params.init_momen;
        end
         
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
        vhWInc = momentum*vhWInc + r/n*(pos_prods-neg_prods) - r*params.wtcost*vhW; % r/n�������1/n����Ϊ֮ǰ��������������������ģ�r����ѧϰ��
        vbInc = momentum*vbInc + (r/n)*(pos_vis_act-neg_vis_act);
        hbInc = momentum*hbInc + (r/n)*(pos_hid_act-neg_hid_act);
         
        invfstd_grad = sum(2*data.*(Ones*vb-data/2)./Fstd,1) + sum(data' .* (vhW*pos_hidprobs') ,2)';
        invfstd_grad = invfstd_grad - ( sum(2*negdata.*(Ones*vb-negdata/2)./Fstd,1) + ...
                                sum( negdata'.*(vhW*neg_hidprobs') ,2 )' );   %���˸��ݣ��������Ǹ���ģ�
                             
        invfstdInc = momentum*invfstdInc + std_rate(epoch)/n*invfstd_grad;
                
        if params.SPARSE == 1 %nair's paper on 3D object recognition
            %update q
            if batch==1 && epoch == 1
                q = mean(pos_hidprobs);
            else
                q_prev = q;
                q = 0.9*q_prev+0.1*mean(pos_hidprobs); %q��ÿ��h1�ı仯����0.1��Ӱ��
            end           
            
            p = params.sparse_p;
            grad = 0.1*params.sparse_lambda/n*sum(pos_hidprobs.*(1-pos_hidprobs)).*(p-q)./(q.*(1-q));
            gradW =0.1*params.sparse_lambda/n*(data'./Fstd'*(pos_hidprobs.*(1-pos_hidprobs))).*repmat((p-q)./(q.*(1-q)), d,1);
             
            hbInc = hbInc + r*grad;
            vhWInc = vhWInc + r*gradW;
        end
         
        ptot = ptot+mean(pos_hidprobs(:)); %��ÿ��H1��ƽ��ֵ������
         
        vhW = vhW + vhWInc;
        vb = vb + vbInc;
        hb = hb + hbInc;    
         
        invfstd = 1./fstd;
        invfstd =  invfstd + invfstdInc;
        fstd = 1./invfstd;
 
        fstd = max(fstd, 0.005); %have a lower bound!        
        %%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end
 
    %if rem(epoch, int32(params.maxepoch/20)) == 0 || epoch < 30
        fprintf(1, ' p%1.2f  ',  ptot/nBatches ); %ptot/nBatches������ÿ�ε�H1��ƽ������ֵ
        fprintf(1, ' error %6.2f  stdr:%.5f fstd(x,y): [%2.3f %2.3f] mm:%.2f ', errsum, std_rate(epoch), fstd(1), fstd(2), momentum);
        fprintf(1, 'vh_W min %2.4f   max %2.4f ', min(min(vhW)), max(max(vhW)));
    %end
    errs(epoch) = errsum;    
end
model.sigma = fstd;
model.W=vhW;
model.b=hb;
model.c=vb;
model.type= 'GB';
model.top=logistic((X./(ones(N,1)*fstd))*vhW + repmat(hb,N,1));