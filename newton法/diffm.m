function y=diffm(x)  %求在x处的梯度
n=length(x);
p=eye(n);
epsilon=1.0e-8;
y=zeros(n,1);
for i=1:n
    f1=fun0(x+epsilon*p(:,i));
    f2=fun0(x-epsilon*p(:,i));
    y(i)=(f1-f2)/(2*epsilon);
end
