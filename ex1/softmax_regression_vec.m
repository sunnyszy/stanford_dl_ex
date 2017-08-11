function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  % m*(k-1), 
  EXP = [exp(transpose(X) * theta) zeros(m,1)];
  % EXP = transpose(EXP);
  EXP_norm = sum(EXP, 2);
  EXP = bsxfun(@rdivide, EXP, EXP_norm);

  I = sub2ind(size(EXP), 1:m, y);
  f = - sum(EXP(I));
  
  
  % g=reshape(g, [num_classes]);
  g=zeros(m, num_classes);
  
  I = sub2ind(size(g), 1:m, y);
  g(I) = 1; 
  g = reshape(g, 1, m, []);
  g = bsxfun(@and, g, ones(n,1,1));
  g = bsxfun(@minus, g, reshape(EXP, 1, m, []));
  g = bsxfun(@times, g, reshape(X, n, m, 1));
  g = - squeeze(sum(g, 2));
  g = g(:, 1:num_classes-1);
  g=g(:); % make gradient a vector for minFunc
 

