function y=deepest(x)
syms x1 x2
tol=1.0e-6;
func=x1^2+2*x2^2;
r0=mdff2(func); 
% r=subs(r0,[x1,x2],x);
z=jacobian(r0);             %����΢��
r=subs(r0,[x1,x2],x);
z=subs(z,[x1,x2],x);
stopc=norm(r);                 %��ֹ����
k=0;
tic;
while k<=10
    if stopc<=tol
    break
    else
        p=-r;            %������˹������
%         alpha=-r'*p/(p'*z*p) ;     %��ȷ���������󲽳�
       alpha=armij(p,x);  
        x=x+p*alpha ;                  %����
        z=jacobian(r0);
        r=subs(r0,[x1,x2],x);           %��һ��������һ��ƫ��
        z=subs(z,[x1,x2],x);           %��һ�������Ķ���ƫ��
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
fprintf('��ȷ���������½�������ֵ������:');
for i=1:1
fprintf('%4d,',y(1,:));
end 
% fprintf(' %4d \n',y(1,2));
fprintf('\n ');
fprintf('һ����������Ϊ k=%4d\n ',k);
fprintf('����ֵΪ y=%4d \n\n\n',y(2,1));