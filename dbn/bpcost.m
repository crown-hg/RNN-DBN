function [ cost, grad ] = bpcost(theta, numhide, numClasses, visibleSize, hiddenSize, lambda, data, labels, topfunc, hidefunc,costtype)
% 根据标签计算误差和梯度

[W,b]=thetatowb(theta, numhide, numClasses, visibleSize, hiddenSize);
a = runnet(data,W,b,numhide,topfunc,hidefunc);
h=a{numhide+2};
y=labels;
[m,nl]=size(data);
squared_error=0.5*sum((h-y).^2,1);
ws = 0;
for i=1:numhide+1
    ws = sum(sum(W{i}.^2));
end
if isequal(costtype,'square')
    cost=1/m*sum(squared_error)+lambda/2*(ws);% 均方误差代价
else
    cost=-1/(m*nl)*sum(sum(y.*log(h)+(1-y).*log(1-h)));%交叉熵代价
end

%% 计算残差
% 这几步比较简化，需要看着推导公式来看
delta=cell(1,numhide+2);
if isequal(costtype,'square')
% 均方误差代价
delta{numhide+2} = -(y-a{numhide+2}).*funcdiff(topfunc, a{numhide+2});
else
delta{numhide+2} = -(y-a{numhide+2}); %交叉熵代价
end

for i=numhide+1:-1:2
    delta{i}=(delta{i+1}*W{i}').*funcdiff(hidefunc,a{i});
%     delta{i}=(delta{i+1}*W{i}').*a{i}.*(1-a{i});
end
Wgrad=cell(1,numhide+1);
bgrad=cell(1,numhide+1);
Wg=[];
bg=[];
for i=1:numhide+1
    Wgrad{i}=1/m*(a{i}'*delta{i+1})+lambda*W{i};
%     Wgrad{i}=1/m*(a{i}'*delta{i+1});
    Wg=[Wg;Wgrad{i}(:)];
    bgrad{i}=1/m*sum(delta{i+1},1);
    bg=[bg;bgrad{i}(:)];
end
grad = [ Wg; bg ];
end