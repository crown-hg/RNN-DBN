function dataPCAwhite = svdpca(data,rv)
%% 程序说明：dataPCAwhite = svdpca(data)，程序中data为 n*T 阶混合数据矩阵，n为信号个数，T为采样点数
% dataPCAwhite为 n*T 阶主分量矩阵。
% n是维数，T是样本数。
if nargin == 0
    error('You must supply the mixed data as input argument.');
end
if length(size(data))>2
    error('Input data can not have more than two dimensions. ');
end
if any(any(isnan(data)))
    error('Input data contains NaN''s.');
end
%% ----------设置参数--------------------
retainVar = rv; %主成分保留的方差
epsilon = 1e-5;

%% --------输出数据的维度和样本数----------
[Dim,NumofSampl] = size(data);
fprintf('Number of signals: %d\n',Dim);
fprintf('Number of samples: %d\n',NumofSampl);
fprintf('Calculate PCA...\n');

%% -----SVD求特征值和特征向量-------
lastEig = Dim;
sigma = data * data' / size(data, 2); %sigma是协方差矩阵
[U,S,Z] = svd(sigma);
eigenValues = diag(S);

%% ———选择主成分的个数lastEig———
eigTotalSum = sum(eigenValues); %所有特征值的和
eigSum = 0;
for i=1:size(eigenValues)
    eigSum = eigSum + eigenValues(i);
    if retainVar <= eigSum/eigTotalSum  %前i个特征值的和比上总和大于设定的保留方差百分比即可
        lastEig = i;
        break;
    end
end
%% ———————选择相应的特征值和特征向量———————

priEigenValues = eigenValues(1:lastEig);
subEigenValues = eigenValues(lastEig+1:size(eigenValues,1));
priEigenVector = U(:,1:lastEig);

%% —————————输出处理的结果信息—————————
fprintf('Selected [%d] dimensions.\n',lastEig);
fprintf('Smallest remaining (non-zero) eigenvalue[ %g ]\n',eigenValues(lastEig));
fprintf('Largest remaining (non-zero) eigenvalue[ %g ]\n',eigenValues(1));
fprintf('Sum of removed eigenvalue[ %g ]\n',sum(subEigenValues));

%% ——————————数据旋转降维白化———————————
dataPCAwhite = diag(1./sqrt(priEigenValues + epsilon)) * priEigenVector'* data;
