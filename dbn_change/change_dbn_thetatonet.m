function net=change_dbn_thetatonet(theta,input_size,hide_size,output_size,delay)
i=input_size;
h=hide_size;
o=output_size;
net=cell(delay,1);
net{1}.U=reshape(theta(1:i*h),i,h);
net{1}.b=reshape(theta(i*h+1:i*h+h),1,h);
for j=2:delay
    offset=(i*h+h*h+h)*(j-2)+i*h+h;
    net{j}.U=reshape(theta(offset+1:offset+i*h),i,h);
    net{j}.W=reshape(theta(offset+i*h+1:offset+i*h+h*h),h,h);
    net{j}.b=reshape(theta(offset+i*h+h*h+1:offset+i*h+h*h+h),1,h);
end
net{delay}.V=reshape(theta(i*h+h+(i*h+h*h+h)*(delay-1)+1:i*h+h+(i*h+h*h+h)*(delay-1)+h*o),h,o);
net{delay}.c=reshape(theta(i*h+h+(i*h+h*h+h)*(delay-1)+h*o+1:i*h+h+(i*h+h*h+h)*(delay-1)+h*o+o),1,o);
end