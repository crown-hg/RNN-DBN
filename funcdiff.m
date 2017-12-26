function diffy = funcdiff( func, y )
% 此函数是得到func函数的一阶导�?% func为原函数，data为输入的x，diffdata为x对应的导�?   
   if isequal(func,@tanh)
       diffy=1-y.^2;
   elseif isequal(func,@logsig)
       diffy=y.*(1-y);
       
   elseif isequal(func,@Softplus)
       diffy=1-exp(-y);
       
   elseif isequal(func,@relu)
       diffy=double(y>0);
       
   elseif isequal(func,@retanh)
       diffy=(1-y.^2)./2;
   
   elseif isequal(func,@flinear)
       diffy=ones(size(y));
       
   elseif isequal(func,@softs)
       diffy=(1.-abs(y)).^2;
   end
end