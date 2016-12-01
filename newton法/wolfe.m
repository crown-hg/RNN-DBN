%利用wolfe条件求alpha
function alpha=wolfe(x,p,c1,c2)
%f:目标函数
%c1:可接受系数1
%c2:可接受系数2
%beta:增大步长倍数
%tol:精度

if ~(c1>0)||~(c1<c2)||~(c2<1)
    error('参数不对');
end
% tol=1.0e-6;
% beta=2;

n=length(x);
x0=x;
alpha=1;
fk=fun0(x);
dfk=diffm(x);
x=x0+alpha*p;
% fk1=Rosen(x,n,a);
% dfk1=mdff(x,a);
b1=0;
b2=inf;
k=0;

while k<=500
    fk1=fun0(x);
    dfk1=diffm(x);
   if  fk-fk1<-c1*alpha*dfk*p'
%     b=alpha;
%     alpha=0.5*(alpha+a);
      if p'*dfk1>=c2*p'*dfk
        break
      else
      b1=alpha;
      alpha=min([2*alpha 0.5*(b1+b2)]);
      x=x0+alpha*p;
      k=k+1;
      end
   else
    b2=alpha;   
    alpha=b1+b2;
    x=x0+alpha*p;
    k=k+1;
   end
end

