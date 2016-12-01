%lowersol
%The L is a lowertriangular matirix
function lowersol=lowersol(L,b)
[m,n]=size(L);
p=length(b);
if(m~=n)
   printf('The L is not a n*n matrix!');
else
    if(p~=m)
        printf('The row of L is not equal to that of b!');
    else
    y(1)=b(1)/L(1,1);
    for i=2:n
        sumy=0;
        for j=i-1:1
            sumy=sumy+L(i,j)*y(j);
        end
        y(i)=(b(i)-sumy)/L(i,i);
    end
    lowersol=y';
    end
end