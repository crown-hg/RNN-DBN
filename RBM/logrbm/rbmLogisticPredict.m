function prediction = rbmLogisticPredict(m, testdata)
ht = logistic(testdata*m.W + repmat(m.b,size(testdata,1),1));
prediction = logistic(ht*m.Wc' + repmat(m.cc,size(ht,1),1));
