x0=[1,1]';
% d0=[-1,-1]';
% alpha=armij(d0,x0 )
 y=newton(x0);    %��ȷ����
 y=newtont(x0);         %wolfe����
 y=newtonamj(x0);          %armijo����
 y=deepest(x0);