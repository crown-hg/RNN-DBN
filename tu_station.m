% 500010092 1 0.0737 0.4719 0.0857 0.4035  501011141
% 500011032 15 0.0551 0.2072 0.0620 0.2183 
% 501012043 101 0.0628 0.1611 0.0563 0.1249
% dr=data.*1956;
% dp=labels.*1956;
load('tu.mat');

re=sum(abs(dp-dr)./dr)/(14*96);

% yr=dr(7*96:1344,15);
% yp=dp(7*96:1344,15);

% yr=dr(1:1344-7*96+1,15);
% yp=dp(1:1344-7*96+1,15);

station=1; % 94 is Station 501011141.
w=1;
d=0;
time1=6;
time2=20;
offoset=2;
if station==1
    titlestring='Station 500010092  ';
else
    titlestring='Station 501011141  ';
end
if w==0
    if d==0
        x=1:(24-time2+time1)*4+1;
        yr=dr(5*96+time2*4-offoset:5*96+(24+time1)*4-offoset,station);
        yp=dp(5*96+time2*4-offoset:5*96+(24+time1)*4-offoset,station);
        titlestring=[titlestring 'Weekend Nighttime'];
        if station==1
            yhigh=400;
        else
            yhigh=800;
        end
    else
        x=1:(time2-time1)*4+1;
        yr=dr(5*96+time1*4-offoset:5*96+time2*4-offoset,station);
        yp=dp(5*96+time1*4-offoset:5*96+time2*4-offoset,station);
        titlestring=[titlestring 'Weekend Daytime'];
        if station==1
            yhigh=1000;
        else
            yhigh=1490;
        end
    end
else
    if d==0
        x=1:(24-time2+time1)*4+1;
        yr=dr(7*96+time2*4-offoset:7*96+(24+time1)*4-offoset,station);
        yp=dp(7*96+time2*4-offoset:7*96+(24+time1)*4-offoset,station);
        titlestring=[titlestring 'Weekday Nighttime'];
        if station==1
            yhigh=400;
        else
            yhigh=800;
        end
    else
        x=1:(time2-time1)*4+1;
        yr=dr(7*96+time1*4-offoset:7*96+time2*4-offoset,station);
        yp=dp(7*96+time1*4-offoset:7*96+time2*4-offoset,station);
        titlestring=[titlestring 'Weekday Daytime'];
        if station==1
            yhigh=1000;
        else
            yhigh=1490;
        end
    end
end

plot(x,yr,'b-',x,yp,'r-');
legend('Observed traffic flow','Predicted traffic flow');
title(titlestring);
grid on;

if d==0
    set(gca,'XTick',1:4:numel(x));
    set(gca,'XTickLabel',{'20', '21','22','23','00','01','02','03','04','05','06'});
else
    set(gca,'XTick',1:4:numel(x));
    set(gca,'XTickLabel',{'06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21'});
end
ylim([0,yhigh]);
xlim([0,numel(x)]);
xlabel('Time(hour)');  
ylabel('Traffic Flow'); 
set(gca,'FontSize',13);
set(gcf,'position',[0,0,500,400]);
set(gca,'position',[0.15,0.14,0.83,0.80]);
