x0=[1,1]';
% d0=[-1,-1]';
% alpha=armij(d0,x0 )
 y=newton(x0);    %¾«È·ËÑË÷
 y=newtont(x0);         %wolfeËÑË÷
 y=newtonamj(x0);          %armijoËÑË÷
 y=deepest(x0);