%求只有两个变量的函数的导数
function y=mdff2(func)
y=zeros(2,1);
syms x1 x2
m=diff(func,x1);
n=diff(func,x2);
y=[m;n];







%{
eps=1.0e-8;
n=length(x);
y=zeros(n,1);
p=eye(n);

for i=1:n                      
    xf=x+eps*p(:,i);
    f1=symf(xf);
    xb=x-eps*p(:,i);
    f2=symf(xb);
    y(i)=(f1-f2)/(2*eps);
end
%}