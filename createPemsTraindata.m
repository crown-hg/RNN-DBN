function [traindata,trainlabels] = createPemsTraindata(daydata,daytimesize,numlink,numperiod)
% numday:总天数
% daytimesize：4个15分钟，96
% numlink：总路数
% numperiod：取几段作为训练数据，4
% numlink = 100;
% daytimesize = 96;
% numperiod=4;
numday=size(daydata,1)/(daytimesize*numlink);
traindata = zeros(numday*daytimesize,numlink*numperiod);
trainlabels = zeros(numday*daytimesize,numlink);
daydata=[daydata;daydata(1:numperiod,:)];
for i=1:numday*daytimesize
    for j=1:numlink
        traindata(i,(j-1)*numperiod+1:j*numperiod)=daydata((j-1)*numday*daytimesize+i:(j-1)*numday*daytimesize+i+numperiod-1,3);
        trainlabels(i,j)=daydata((j-1)*numday*daytimesize+i+numperiod,3);
    end
end