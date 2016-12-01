function alpha=armij(d0,x0)
% Armijo step
alpha=1;
pho=1/2;
sigma=0.5;
itermax=10;
d=d0;
for i=1:itermax
    x1=x0+alpha*d;
    f1=fun0(x1);
    f0=fun0(x0);
    df=fund(x0)'*d;
    if f1-f0>sigma*alpha*df
       alpha=pho*alpha;
    else
        alpha=alpha;
        break
    end
end
end

