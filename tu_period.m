load('tu.mat');
offoset=-1;
station=1;
w=1;
if w==1
    x=7*96+offoset:9*96+offoset; % weekday
else
    x=5*96+offoset:7*96+offoset; % weekend
end
if station==1
    yhigh=1000;
    textHigh=900;
else
    yhigh=1600;
    textHigh=1500;
end
y=dr(x,station);
plot(1:96*2+1,y,'b-');
hold on;
plot([25 25],[0 yhigh],'r-',[81 81],[0 yhigh],'r-',[121 121],[0 yhigh],'r-',[177 177],[0 yhigh],'r-');
set(gcf,'position',[0,0,800,380]);
set(gca,'position',[0.1,0.16,0.88,0.82]);
set(gca,'XTick',1:8:96*2+1);
set(gca,'XTickLabel',{'00','02','04','06','08','10','12','14','16','18','20','22','00','02','04','06','08','10','12','14','16','18','20','22','24'});
legend('Observed traffic flow');
set(gca,'FontSize',15);
text(42,textHigh,'Daytime','FontSize',15);
text(92,textHigh,'Nighttime','FontSize',15);
text(138,textHigh,'Daytime','FontSize',15);
xlabel('Time(hour)');
ylabel('Traffic Flow');
ylim([0,yhigh]);
grid on;