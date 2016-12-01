function net=rnn_thetatonet(theta,input_size,hide_size,output_size)
i = input_size;
h=hide_size;
o=output_size;
net.U=reshape(theta(1:i*h), i, h);
net.V=reshape(theta(i*h+1:i*h+h*o), h, o);
net.W=reshape(theta(i*h+h*o+1:i*h+h*o+h*h), h, h);
net.b=reshape(theta(i*h+h*o+h*h+1:i*h+h*o+h*h+h), 1, h);
net.c=reshape(theta(i*h+h*o+h*h+h+1:i*h+h*o+h*h+h+o), 1, o);
end