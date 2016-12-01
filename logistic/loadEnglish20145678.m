%numday:天数
%daytimesize：多少个15分钟，96
%numlink：多少条路，50
%numperiod：取几段作为训练数据，4
clc;
clear;
tic;
conna=database('mydb','','');
daytimesize=96;
numlink=200;
numperiod=4;
date30 = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'};
date31 = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'};
datamonth=cell(1,12);
for month=5:12
    if month==6||month==9||month==11
        date=date30;
    else
        date=date31;
    end
    numday=size(date,2);
    data = zeros(daytimesize * numlink,3,numday);
    dt = zeros(daytimesize * numlink,3);
    for i=1:numday
        str=sprintf('select [LinkRef],[TimePeriod],[Flow] from [test].[dbo].[2014%s] where [LinkRef]in (SELECT top %d [LinkRef] FROM [test].[dbo].[201405] group by [LinkRef] order by AVG([Flow]) desc) and [Date]=''2014-%s-%s 00:00:00.00''',date{month},numlink,date{month},date{i});
        curs=exec(conna,str);
        cursf = fetch(curs);
        d = cursf.data;
        d = sortrows(d,1);
        n=1;
        dt = zeros(daytimesize*numlink,3);
        for k=1:daytimesize*numlink
            dt(k,1) = n;
            if mod(k,daytimesize)==0
                n = n+1; 
            end
            dt(k,2)=d{k,2}+1;
            dt(k,3)=d{k,3}; 
        end
        dt = sortrows(dt,2); dt = sortrows(dt,1);
        data(:,:,i)=dt;
    end
    % 归一化处理
    if month==5
        linkid=unique(d(:,1));
        datarow=zeros(1,daytimesize*numlink*numday);
        for i=1:numday
            datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)=data(:,3,i)';
        end
        [datarow,ps]=mapminmax(datarow,0,1);
        for i=1:numday
            data(:,3,i)=datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)';
        end
    else
        datarow=zeros(1,daytimesize*numlink*numday);
        for i=1:numday
            datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)=data(:,3,i)';
        end
        datarow=mapminmax('apply',datarow,ps);
        for i=1:numday
            data(:,3,i)=datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)';
        end
    end
    datamonth{month}=data; 
end
%save English_2014_5678_link200 datamonth ps linkid;
train.data=zeros((31+30+31)*daytimesize,numlink*numperiod);
train.labels=zeros((31+30+31)*daytimesize,numlink);
for month=5:12
    daydata=datamonth{month};
    numday=size(datamonth{month},3);
    if month==12
        traindata = zeros((numday-1)*daytimesize,numlink*numperiod);
        trainlabels = zeros((numday-1)*daytimesize,numlink);
    else
        traindata = zeros(numday*daytimesize,numlink*numperiod);
        trainlabels = zeros(numday*daytimesize,numlink);
    end
    for w=1:numday
        if month==12&&w==numday
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
        train.data((31+30+31)*daytimesize+1:(31+30+31+31)*daytimesize,:)=traindata;
        train.labels((31+30+31)*daytimesize+1:(31+30+31+31)*daytimesize,:)=trainlabels;
        case 9
        train.data((31+30+31+31)*daytimesize+1:(31+30+31+31+30)*daytimesize,:)=traindata;
        train.labels((31+30+31+31)*daytimesize+1:(31+30+31+31+30)*daytimesize,:)=trainlabels;
        case 10
        train.data((31+30+31+31+30)*daytimesize+1:(31+30+31+31+30+31)*daytimesize,:)=traindata;
        train.labels((31+30+31+31+30)*daytimesize+1:(31+30+31+31+30+31)*daytimesize,:)=trainlabels;
        case 11
        train.data((31+30+31+31+30+31)*daytimesize+1:(31+30+31+31+30+31+30)*daytimesize,:)=traindata;
        train.labels((31+30+31+31+30+31)*daytimesize+1:(31+30+31+31+30+31+30)*daytimesize,:)=trainlabels;
        case 12
        test.data=traindata;
        test.labels=trainlabels;
    end
end
save train5_11test12 train test ps linkid;
toc;