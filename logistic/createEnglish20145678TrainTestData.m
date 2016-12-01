%numday:天数，20
%daytimesize：多少个15分钟，96
%numlink：多少条路，50
%numperiod：取几段作为训练数据，4
daytimesize=96;
daytime=56;
nighttime=40;
daybegin=25;
dayend=80;
numlink=200;
numperiod=4;
addpath data/
load('English_2014_5678_link200.mat');
train.data=zeros((31+30+31)*daytimesize,numlink*numperiod);
train.labels=zeros((31+30+31)*daytimesize,numlink);
for month=5:8
    daydata=datamonth{month};
    numday=size(datamonth{month},3);
    if month==8
        traindata = zeros((numday-1)*daytimesize,numlink*numperiod);
        trainlabels = zeros((numday-1)*daytimesize,numlink);
    else
        traindata = zeros(numday*daytimesize,numlink*numperiod);
        trainlabels = zeros(numday*daytimesize,numlink);
    end
    for w=1:numday
        if month==8&&w==numday
            break;
        end
        for l=0:numlink-1
            for i=l*daytimesize+1:l*daytimesize+daytimesize
                for k=i:i+numperiod-1
                    if k>l*daytimesize+daytimesize
                        if w<numday
                            traindata(daydata(i,2,w)+(w-1)*daytimesize,k-i+1+l*numperiod) = daydata(k-daytimesize,3,w+1);
                        else
                            traindata(daydata(i,2,w)+(w-1)*daytimesize,k-i+1+l*numperiod) = datamonth{month+1}(k-daytimesize,3,1);
                        end
                    else
                        traindata(daydata(i,2,w)+(w-1)*daytimesize,k-i+1+l*numperiod) = daydata(k,3,w);
                    end
                end
                k = k+1;
                n = k;
                if k>l*daytimesize+daytimesize
                    n = k-(l*daytimesize+daytimesize);
                end
                trainlabels(daydata(i,2,w)+(w-1)*daytimesize,l+1)=daydata(n,3,w);
            end
        end
    end
    switch month
        case 5
        train.data(1:31*daytimesize,:)=traindata;
        train.labels(1:31*daytimesize,:)=trainlabels;
        case 6
        train.data(31*daytimesize+1:(31+30)*daytimesize,:)=traindata;
        train.labels(31*daytimesize+1:(31+30)*daytimesize,:)=trainlabels;
        case 7
        train.data((31+30)*daytimesize+1:(31+30+31)*daytimesize,:)=traindata;
        train.labels((31+30)*daytimesize+1:(31+30+31)*daytimesize,:)=trainlabels;
        case 8
        test.data=traindata;
        test.labels=trainlabels;
    end
end
save train567test8 train test;