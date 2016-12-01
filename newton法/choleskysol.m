%cholesky 解Ax=b,其中A为对称正定
function choleskysol=choleskysol(A,b)
[m,n]=size(A);
L=zeros(m,n);
L(1,1)=sqrt(A(1,1));
for j=2:n
    L(j,1)=A(j,1)/L(1,1);
end
for i=2:n
    for j=i:m
        p(j,i)=0;
        for k=1:i-1
            p(j,i)=p(j,i)+L(j,k)*L(i,k);
            t(j,i)=A(j,i)-p(j,i);
            if(j==i)
                L(j,i)=sqrt(t(j,i));
            else
            L(j,i)= t(j,i)/L(i,i);
            end
        end
    end
end
y=lowersol(L,b); %解Ly=b
x=uppersol(L',y);  %解L'x=y
choleskysol=x;




