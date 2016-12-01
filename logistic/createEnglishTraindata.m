function [traindata,trainlabels] = createEnglishTraindata(filename, numday,daytimesize,numlink,numperiod)
%numday:������20
%daytimesize�����ٸ�15���ӣ�96
%numlink��������·��50
%numperiod��ȡ������Ϊѵ�����ݣ�4
addpath data/
load(filename);
traindata = zeros(numday*daytimesize,numlink*numperiod);
trainlabels = zeros(numday*daytimesize,numlink);
for w=1:numday
    for l=0:numlink-1
        for i=l*daytimesize+1:l*daytimesize+daytimesize
            for k=i:i+numperiod-1
                if k>l*daytimesize+daytimesize
                    traindata(daydata(i,2,w)+(w-1)*daytimesize,k-i+1+l*numperiod) = daydata(k-daytimesize,3,w+1);
                else
                    traindata(daydata(i,2,w)+(w-1)*daytimesize,k-i+1+l*numperiod) = daydata(k,3,w);
                end
            end
            k = k+1;
            n = k;
            if k>l*daytimesize+daytimesize
                n = k-(l*daytimesize+daytimesize);
            end
            trainlabels(daydata(i,2,w)+(w-1)*daytimesize,l+1)=daydata(n,3,w);
        end
    end
end