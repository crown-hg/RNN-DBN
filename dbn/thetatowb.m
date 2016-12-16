function [ Weight, bias ] = thetatowb( theta, numhide, numClasses, visibleSize, hiddenSize )
% 把theta转化成权值W和b

W=cell(1,numhide+1);
b=cell(1,numhide+1);
hv=hiddenSize*visibleSize;
hh=hiddenSize*hiddenSize;
hn=hiddenSize*numClasses; 
% hn=visibleSize*numClasses;%##
for i=1:numhide
    if i==1
        W{i}=reshape(theta(1:hv), visibleSize, hiddenSize);
    else
        W{i}=reshape(theta(hv+(i-2)*hh+1:hv+(i-1)*hh), hiddenSize, hiddenSize);
    end
end
W{numhide+1}=reshape(theta(hv+(numhide-1)*hh+1:hv+(numhide-1)*hh+hn),hiddenSize, numClasses); 
% W{numhide+1}=reshape(theta(hv+(numhide-1)*hh+1:hv+(numhide-1)*hh+hn), visibleSize, numClasses); %##
for j=1:numhide
    b{j}=reshape(theta(hv+(numhide-1)*hh+hn+(j-1)*hiddenSize+1:hv+(numhide-1)*hh+hn+j*hiddenSize), 1, hiddenSize);
end
b{numhide+1} = reshape(theta(hv+(numhide-1)*hh+hn+numhide*hiddenSize+1:hv+(numhide-1)*hh+hn+numhide*hiddenSize+numClasses), 1, numClasses);
Weight = W;
bias = b;
end

