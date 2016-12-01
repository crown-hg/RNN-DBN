function y=deepest(x)
syms x1 x2
tol=1.0e-6;
func=x1^2+2*x2^2;
r0=mdff2(func); 
% r=subs(r0,[x1,x2],x);
z=jacobian(r0);             %二阶微分
r=subs(r0,[x1,x2],x);
z=subs(z,[x1,x2],x);
stopc=norm(r);                 %终止条件
k=0;
tic;
while k<=10
    if stopc<=tol
    break
    else
        p=-r;            %用乔列斯基求方向
%         alpha=-r'*p/(p'*z*p) ;     %精确线性搜索求步长
       alpha=armij(p,x);  
        x=x+p*alpha ;                  %迭代
        z=jacobian(r0);
        r=subs(r0,[x1,x2],x);           %下一步迭代的一阶偏导
        z=subs(z,[x1,x2],x);           %下一步迭代的二阶偏导
%         qqq=subs(r);
%         stopc=norm(qqq);
       stopc=norm(r);
        k=k+1;
    end
end
toc;
time=toc;
y(1,:)=x';
y(2,1)=subs(func,[x1,x2],x);
y(3,1)=k;
y(4,1)=time;
fprintf('精确搜索最速下降法最优值点如下:');
for i=1:1
fprintf('%4d,',y(1,:));
end 
% fprintf(' %4d \n',y(1,2));
fprintf('\n ');
fprintf('一共迭代部数为 k=%4d\n ',k);
fprintf('最优值为 y=%4d \n\n\n',y(2,1));