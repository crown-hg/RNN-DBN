function x = newton(x0,tol)
if nargin < 2
    tol = 1.0e-5;
end
x = x0 - fun(x0)/dfun(x0);
n = 1; 
while (norm(x-x0)>tol) && (n<1000)
    x0 = x;
    x = x0 - fun(x0)/dfun(x0);
    n = n + 1; 
end
n