clear;
% numday:����10
% daytimesize�����ٸ�15���ӣ�daytimesize
% numlink��������·��numlink
conn=database('pems','root','root','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/pems');
numday = 92;
numlink = 198;
daytimesize = 96;
d=cell(numday*daytimesize*3*numlink,3);
% ��stationid
for i=1:numday
    tic;
    s1 = sprintf('SELECT Time,Station,Totalflow from d05_2016_060708 LIMIT %d,%d',(i-1)*daytimesize*3*numlink,daytimesize*3*numlink); 
    cursf = fetch(exec(conn,s1));
    d((i-1)*daytimesize*3*numlink+1:i*daytimesize*3*numlink,:) = cursf.Data;
    fprintf('第%d天完成\n',i);
    toc;
end
stationid = d(1:numlink,2);
d = sortrows(d,[2 1]);
daydata =zeros(daytimesize*numlink*numday,3);
for i=1:daytimesize*numlink*numday
    daydata(i,1) = d{(i-1)*3+1,2};
    daydata(i,2) = mod(i,daytimesize);
    if mod(i,daytimesize)==0
        daydata(i,2) = daytimesize;
    end
    daydata(i,3) = d{(i-1)*3+1,3}+d{(i-1)*3+2,3}+d{(i-1)*3+3,3};
end
datarow=daydata(:,3)';
[datarow,ps]=mapminmax(datarow,0,1);
daydata(:,3)=datarow';
save pemsd05_2016060708_day92_link147 daydata ps stationid;