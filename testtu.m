fig1 = figure(1);

% ���û�ͼ���ڵ�λ�úʹ�С: [left, bottom, width, height], ȱʡ������Ϊ��λ
% ��ͨ�� 'Unit' ���õ�λ: inches | centimeters | normalized | points | pixels| characters
% ��: set(fig11,'Unit','centimeters'])  
set(fig1,'Position',[100,500,500,100])  

% % ������ͼ�����λ�����С: [left bottom width height], ȱʡ��λ�� normalized
% % ��ͨ�� 'Unit' ���õ�λ: inches | centimeters | normalized | points | pixels| characters
% % ��: axes1=axes('Parent',fig11,'Unit','pixels','Position',[0 0 50 50])
axes1 = axes('Parent',fig1,'Position',[0.05 0.52 0.43 0.43]);
imshow(axes1,[0,255]);  
text(30,-10, '(a) Original image');
% 
% % ����������ͼ
% axes1 = axes('Parent',fig11,'Position',[0.50 0.52 0.43 0.43]);
% imshow(fn,[0,255]); 
% text(30,-10, '(b) Observed image');
% 
% axes1 = axes('Parent',fig11,'Position',[0.05 0.01 0.43 0.43]);
% imshow(x0n,[0,255]); 
% text(30,-8, ['(c) Initial guess']);
% 
% axes1 = axes('Parent',fig11,'Position',[0.50 0.01 0.43 0.43]);
% imshow(x22n,[0,255]); 
% text(30,-8, '(d) Restored image');
