clear;
clc;
bpcost=@tanhbpcost;
rbmFit=@rbmFittanh;
rbm=@rbmBB;
run=@tanhrunnet;
tanh20161017=[];
numday=362;
daytimesize=96;
numlink=100;
addpath data/
load('pemsd05_2013_day363_link100.mat');
%获取当前时间，把时间和data写入log
for numperiod = 5:-1:4
    [data,labels] = createPemsTraindata(daydata,daytimesize,numlink,numperiod);
    for layer = 1:1
        for hidenodenum = 100:100:500
        hidelayer=[];
            for h=1:layer
            hidelayer = [hidelayer,hidenodenum];
            end
            for week=1:1
                for day =1:1
                    [cost,time,MRE,MAE,RMSE]=tanhdbn(data,labels,numperiod,week,day,hidelayer,bpcost,rbmFit,rbm,run,ps);
                    r=[numperiod,layer,hidenodenum,week,day,cost,MRE,MAE,RMSE,time];
                    tanh20161017=[tanh20161017;r];
                end
            end
        end
    end
end