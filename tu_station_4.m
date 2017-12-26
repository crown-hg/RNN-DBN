fig1=subplot(141);
set(fig1,'position',[0.04,0.15,0.20,0.78]);
station = 94;
subFig(station,1,1);

fig2=subplot(142);
set(fig2,'position',[0.29,0.15,0.20,0.78]);
subFig(station,1,0);

fig3=subplot(143);
set(fig3,'position',[0.54,0.15,0.20,0.78]);
subFig(station,0,1);

fig4=subplot(144);
set(fig4,'position',[0.79,0.15,0.20,0.78]);
subFig(station,0,0);

set(gcf,'position',[0,0,1600,320]);

function subFig(station,w,d)
load('tu.mat');
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
        offoset=-6;
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
        offoset=-3;
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
    set(gca,'XTickLabel',{'20','21','22','23','00','01','02','03','04','05','06'});
else
    set(gca,'XTick',1:4:numel(x));
    set(gca,'XTickLabel',{'06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21'});
end
ylim([0,yhigh]);
xlim([0,numel(x)]+1);
xlabel('Time(hour)');
ylabel('Traffic Flow');
end