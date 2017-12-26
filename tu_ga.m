% [vx,vmfit]=ga_main();
% 
% % w1d1 0.8976,0.9305 0.8206,0.9002
% % w1d0 0.7445,0.8002 0.6546,0.7550
% % w0d1 0.8372,0.9117 0.6546,0.7550
% % w0d0 0.7188,0.7950 0.6521,0.7460
% 
% vxd=mapminmax(vx,0.7188,0.7950);
% vmfitd=mapminmax(vmfit,0.6521,0.7460);
load('gaw0d0');
plot(vxd,'b');
hold on;
plot(vmfitd,'r');
title('Weekend Nighttime');
legend('Highest accuracy','Average accuracy');
xlabel('Generations');ylabel('Accuracy');hold on;
grid on;
ylim([0.6,0.95]);
set(gcf,'position',[0,0,400,320]);
set(gca,'position',[0.13,0.14,0.81,0.80]);

% save gaw0d0 vmfitd vxd