clear;
clc;
%numday:天数，10
%daytimesize：多少个15分钟，96
%numlink：多少条路，50
%numperiod：取几段作为训练数据，4
conna=database('mydb','','');
daytimesize=96;
numlink=200;
date = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'};
numday=size(date,2);
daydata =zeros(daytimesize*numlink,3,numday);
dt = zeros(daytimesize*numlink,3);
for i=1:numday
    str=sprintf('select [LinkRef],[TimePeriod],[Flow] from [test].[dbo].[201405] where [LinkRef]in (SELECT top %d [LinkRef] FROM [test].[dbo].[201405] group by [LinkRef] order by AVG([Flow]) desc) and [Date]=''2014-05-%s 00:00:00.00''',numlink,date{i});
    curs=exec(conna,str);
    cursf = fetch(curs);
    d = cursf.data;
    d=sortrows(d,1);
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
    daydata(:,:,i)=dt;
end
% 归一化处理
datarow=zeros(1,daytimesize*numlink*numday);
for i=1:numday
    datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)=daydata(:,3,i)';
end
[datarow,ps]=mapminmax(datarow,0,1);
for i=1:numday
    daydata(:,3,i)=datarow((i-1)*daytimesize*numlink+1:i*daytimesize*numlink)';
end
save data01_201405_link200 daydata ps;