%初始化
clear
s=0.01;
Yout(1,1)=3.9;
Yout(2,1)=4.5;
Yout(3,1)=4.5;
Rin=5.5;
U0=18;
cha0=0;
Ic=eye(3);
Ii=eye(25);
Ec0=2;
Ei0=2;
g=4710;
Ch0=10^(-4);
e(1,1)=1.6;
e(2,1)=1.5;
Ch0=10^(-4.1);
xc0=[0;18.4;0];
netc0=[0;0;0];
c=10;
hi=0.001;
hc=0.001;
Q=1.05;
xwc10=[0,0,0]';
xwc11=[0.1,4.8,0.5]';
tiaocha=0;
aef=0;
z=10;
y=0.03;
y1=0.6;
a=1;
b=1;
r=0.2;
max=300;
Wij=y*ones(5,4);
W1i=y1*ones(1,5);
W13=(xwc11)';
%计算网络输出及误差指标函数E(x)
for k=3:1:max
    %NNC输出
   e(k,1)=Rin-Yout(k,1);
   P=e(k,1);
   I=e(k,1);
   D=e(k,1)-e((k-1),1);
   netc=[P;I;D];
   %比例元
     if (netc(1,1)<-1)
        xc(1,1)=-1;
    elseif (netc(1,1)>1)
        xc(1,1)=1;
    else 
        xc(1,1)=netc(1,1);
    end
    %积分元
    if (abs(netc(2,1))<=a)
        fnetc=1;
    elseif (abs(netc(2,1))>=(a+b))
        fnetc=abs(netc(2,1))/(2*(a+b));
    else
        if((netc(2,1)<0))
        fnetc=abs(netc(2,1))/(a+b);
    else(netc(2,1)>0)
        fnetc=(a-abs(netc(2,1))+b)/a;
    end
    end
        xc(2,1)=xc0(2,1)+fnetc*netc(2,1);
       if (xc(2,1)>10)
          xc(2,1)=10;
        elseif (xc(2,1)<2)
           xc(2,1)=2;
       else
          xc(2,1)=xc(2,1);
        end
     %微分元
     xc(3,1)=(1-r)*(netc(3,1))+r*xc0(3,1);
     if (xc(3,1)>1)
         xc(3,1)=1;
     elseif (xc(3,1)<-1)
         xc(3,1)=-1;
     else 
         xc(3,1)=xc(3,1);
     end
     xu=W13*xc/z;
     U=xu;
  if(U<4)
       U=4;
   elseif(U>22)
       U=22;
   end
     
     %NNI输出
     xir=[Yout(k,1);Yout((k-1),1);U;U0];
     neti=Wij*xir;
     for i=1:1:5
         xi(i,1)=tanh(neti(i,1));
     end
     Ymout(k,1)=W1i*xi;
     %修正Ymout错误值
     if (Ymout(k,1)>7)
         Ymout(k,1)=7;
      
     elseif (Ymout(k,1)<2)
         Ymout(k,1)=2;
        
     end
     Ec=(e(k,1))^2;
     Ei=(Yout(k,1)-Ymout(k,1))^2;
     %if(Ec<s)
      %   break;
      %end
     if (Ec<Ec0)
         hc=hc/Q;
     else
         hc=hc*Q;
     end
     if (Ei<Ei0)
         hi=hi/Q;
     else
         hi=hi*Q;
     end
%计算Jacobian矩阵J(x)
xwi=[Wij(1,1),Wij(1,2),Wij(1,3),Wij(1,4),Wij(2,1),Wij(2,2),Wij(2,3),Wij(2,4),Wij(3,1),Wij(3,2),Wij(3,3),Wij(3,4),Wij(4,1),Wij(4,2),Wij(4,3),Wij(4,4),Wij(5,1),Wij(5,2),Wij(5,3),Wij(5,4),W1i(1,1),W1i(1,2),W1i(1,3),W1i(1,4),W1i(1,5)]';
%NNI
for i1=21:1:25
    Ji(1,i1)=-xi((i1-20),1);
end
momenti=1;
for i2=1:1:5
    for j2=1:1:4
        Ji(1,momenti)=-W1i(1,i2)*(1-xi(i2,1)^2)*xir(j2,1);
        momenti=momenti+1;
    end
end
%NNC
Yout_U=0;
xwc11=W13';
for i3=1:1:5
    Yout_U0=(1-xi(i3,1)^2)*W1i(1,i3)*Wij(i3,3);
    Yout_U=Yout_U+Yout_U0;
end
for i4=1:1:3
    Jc(1,i4)=(-(xc(i4,1)*Yout_U)/z);
end
%分别计算修正后的权值，并通过权值修正来计算U及Yout
chakanbianhua=inv((Jc'*Jc+hc*Ic))*Jc'*e(k,1)
xwc1=xwc11-chakanbianhua+aef*(xwc11-xwc10);
xwi1=xwi-inv((Ji'*Ji+hi*Ii))*Ji'*(Yout(k,1)-Ymout(k,1));
xwc10=xwc11;
xwc11=xwc1;
for i5=21:1:25
    W1i(1,(i5-20))=xwi1(i5,1);
end
momenti1=1;
for i6=1:1:5
    for j6=1:1:4
        Wij(i6,j6)=xwi1(momenti1,1);
        momenti1=momenti1+1;
    end
end
    W13=xwc11';
%需进行循环的量
Ec0=Ec;Ei0=Ei;netc0=netc;xc0=xc;
%计算修正后的U,Yout值
   xu=W13*xc/z;
     U=xu;
   %权值峰值的限制
   if(U<4)
       U=4;
   elseif(U>22)      
       U=22;
   end
%防止误差过大的办法
if((Rin-Yout(k,1))>2)
 U=U+tiaocha;
 %else
 %if((Rin-Yout(k,1))<-1)
  %U=U-tiaocha;  
end
%修正后再进行循环的量
U0=U;
%实际的输出,
cha=cha0+(2*(g-250*U))/(30000000*pi);
if (cha>10^(-2))
    cha=10^(-2)-Ch0;    
end
if((Ch0==10^(-2))&(cha>0))
    cha=0;
end
Ch=Ch0+cha;
if (Ch<10^(-7))
    cha0=Ch-10^(-7);
    Ch=10^(-7);
elseif(Ch>10^(-2))
  Ch=10^(-2);
   cha0=0;
else
   cha0=0;
end
Ch0=Ch;
Yout(k+1,1)=-log10(Ch);
end
t=1:1:k;
plot(t,Ymout(t,1),'b*',t,Yout(t,1),'r+',t,e(t,1),'mx');