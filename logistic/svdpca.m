function dataPCAwhite = svdpca(data,rv)
%% ����˵����dataPCAwhite = svdpca(data)��������dataΪ n*T �׻�����ݾ���nΪ�źŸ�����TΪ��������
% dataPCAwhiteΪ n*T ������������
% n��ά����T����������
if nargin == 0
    error('You must supply the mixed data as input argument.');
end
if length(size(data))>2
    error('Input data can not have more than two dimensions. ');
end
if any(any(isnan(data)))
    error('Input data contains NaN''s.');
end
%% ----------���ò���--------------------
retainVar = rv; %���ɷֱ����ķ���
epsilon = 1e-5;

%% --------������ݵ�ά�Ⱥ�������----------
[Dim,NumofSampl] = size(data);
fprintf('Number of signals: %d\n',Dim);
fprintf('Number of samples: %d\n',NumofSampl);
fprintf('Calculate PCA...\n');

%% -----SVD������ֵ����������-------
lastEig = Dim;
sigma = data * data' / size(data, 2); %sigma��Э�������
[U,S,Z] = svd(sigma);
eigenValues = diag(S);

%% ������ѡ�����ɷֵĸ���lastEig������
eigTotalSum = sum(eigenValues); %��������ֵ�ĺ�
eigSum = 0;
for i=1:size(eigenValues)
    eigSum = eigSum + eigenValues(i);
    if retainVar <= eigSum/eigTotalSum  %ǰi������ֵ�ĺͱ����ܺʹ����趨�ı�������ٷֱȼ���
        lastEig = i;
        break;
    end
end
%% ��������������ѡ����Ӧ������ֵ������������������������

priEigenValues = eigenValues(1:lastEig);
subEigenValues = eigenValues(lastEig+1:size(eigenValues,1));
priEigenVector = U(:,1:lastEig);

%% �������������������������Ľ����Ϣ������������������
fprintf('Selected [%d] dimensions.\n',lastEig);
fprintf('Smallest remaining (non-zero) eigenvalue[ %g ]\n',eigenValues(lastEig));
fprintf('Largest remaining (non-zero) eigenvalue[ %g ]\n',eigenValues(1));
fprintf('Sum of removed eigenvalue[ %g ]\n',sum(subEigenValues));

%% ��������������������������ת��ά�׻�����������������������
dataPCAwhite = diag(1./sqrt(priEigenValues + epsilon)) * priEigenVector'* data;
