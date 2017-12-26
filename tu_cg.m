load('scgw0d0');

plot(1:40,sd(1:200:8000,1),'g.-',1:40,lbfgs(1:8:320,1),'b.-',1:40,scg(1:400:16000,1),'r.-');
legend('GD','LBFGS','FR-CG');
% set(gca,'XTick',5:5:40);
% set(gca,'XTickLabel',{'1000','2000','3000','4000','5000','6000','7000','8000'});
set(gca,'XTick',10:10:40);
set(gca,'XTickLabel',{'1000','2000','3000','4000'});

xlabel('Fine-tuning Iteration');
ylabel('MRE of Test Set');
grid on;
set(gcf,'position',[0,0,400,320]);
set(gca,'position',[0.13,0.14,0.83,0.84]);

% ylim([0.15,0.5]);
% hold on;
% for i=1:2:40
%     plot(i,sd((i-1)*100+1),'g.',i,lbfgs((i-1)*100+1),'b.',i,scg((i-1)*100+1),'r.');
%     hold on;
% end