%uppersol
%solve the L*y=b where L a uppertriangular matirix
function uppersol=uppersol(L,b)
[m,n]=size(L);
p=length(b);
if(m~=n)
   printf('The L is not a n*n matrix!');
else
    if(p~=m)
        printf('The row of L is not equal to that of b!');
    else
    x(n)=b(n)/L(n,n);
    for i=n-1:-1:1
        sumx=0;
        for j=i+1:n
            sumx=sumx+L(i,j)*x(j);
        end
        x(i)=(b(i)-sumx)/L(i,i);
    end
    uppersol=x';
    end
end



