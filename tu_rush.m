load('tu.mat');
offoset=1;
station=1;
w=1;
time1=6;
time2=20;
if station==1
    yhigh=1000;
    textHigh=900;
else
    yhigh=1600;
    textHigh=1500;
end
x=1:(time2-time1)*4+1;
y=dr(7*96+time1*4-offoset:7*96+time2*4-offoset,station);
plot(x,y,'b-');
hold on;
plot([5 5],[0 yhigh],'r-',[9 9],[0 yhigh],'r-',[43 43],[0 yhigh],'r-',[47 47],[0 yhigh],'r-');
set(gcf,'position',[0,0,800,380]);
set(gca,'position',[0.1,0.16,0.88,0.82]);
set(gca,'XTick',1:4:96*2+1);
set(gca,'XTickLabel',{'06','07','08','09','10','11','12','13','14','15','16','17','18','19','20'});

legend('Observed traffic flow');
text(25,textHigh,'Daytime','FontSize',15);
text(10,textHigh,'Peak time','FontSize',15);
text(15,textHigh,'Peak time','FontSize',15);
set(gca,'FontSize',15);
xlabel('Time(hour)');
ylabel('Traffic Flow');
ylim([0,yhigh]);
grid on;