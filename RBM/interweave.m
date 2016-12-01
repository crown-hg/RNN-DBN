function C = interweave(A, B)
% Combine two cell arrays into one, alternating elements from each
% 这段代码就是为了把A，B中的值一一对应起来
% 比如A={'name','age'},B={'micheal',3}
% 得到结果C={'name','micheal','age',3}
% * C is a row vector
% * Empty elements are removed
% * If length(A) ~= length(B), the remaining elements are added to the end.

% This file is from matlabtools.googlecode.com

A = A(:);
B = B(:);
nA = numel(A);
nB = numel(B);
C = cell(1, nA+nB);
C(1:2:2*nA-1) = A;
C(2:2:2*nB) = B;
%C = removeEmpty(C);

end
