function [ data_s,md,sd ] = initialdata( data )
% ���ݹ�һ��
% data��n*T��
% n��ά�ȣ�T��������

md = mean(data,1); % ������ֵ
datat = bsxfun(@minus,data,md);
sd = std(datat,0,1); % ������׼��
sd(sd==0)=1;
data_s = bsxfun(@rdivide,datat,sd);
end

