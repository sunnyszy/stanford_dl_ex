function [ h ] = softmax( a )
%softmax function
%   input: a k*m
%   output: h k*m
a = exp(a);
norm = sum(a, 1);
h = bsxfun(@rdivide, a, norm);

end

