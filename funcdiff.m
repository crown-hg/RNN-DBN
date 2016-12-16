function diffdata = funcdiff( func, data )
% 此函数是得到func函数的一阶导数
% func为原函数，data为输入的x，diffdata为x对应的导数
   if isequal(func,@tanh)
       diffdata=1-data.^2;
       
   elseif isequal(func,@logsig)
       diffdata=data.*(1-data);
       
   elseif isequal(func,@Softplus)
       diffdata=1-exp(-data);
       
   elseif isequal(func,@relu)
       diffdata=double(data>0);
       
   elseif isequal(func,@retanh)
       diffdata=(1-data.^2)./2;
   
   elseif isequal(func,@flinear)
       diffdata=ones(size(data));
   end
end