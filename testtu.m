fig1 = figure(1);

% 设置绘图窗口的位置和大小: [left, bottom, width, height], 缺省以像素为单位
% 可通过 'Unit' 设置单位: inches | centimeters | normalized | points | pixels| characters
% 如: set(fig11,'Unit','centimeters'])  
set(fig1,'Position',[100,500,500,100])  

% % 设置子图的相对位置与大小: [left bottom width height], 缺省单位是 normalized
% % 可通过 'Unit' 设置单位: inches | centimeters | normalized | points | pixels| characters
% % 如: axes1=axes('Parent',fig11,'Unit','pixels','Position',[0 0 50 50])
axes1 = axes('Parent',fig1,'Position',[0.05 0.52 0.43 0.43]);
imshow(axes1,[0,255]);  
text(30,-10, '(a) Original image');
% 
% % 绘制其它子图
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
