function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
Act = cell(numHidden+2, 1);  %# act in all layers
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
nl = numHidden+2;
z = cell(nl, 1);
err = cell(nl, 1);
[n,m] = size(data);


Act{1} = data;
for i = 2:nl
    z{i} = bsxfun(@plus, stack{i-1}.W * Act{i-1}, stack{i-1}.b);
    if i < nl
        Act{i} = sigmoid(z{i});
    else
        Act{i} = softmax(z{i});
    end
end
pred_prob = Act{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ceCost = log(pred_prob); 
I = sub2ind(size(ceCost), labels', 1:numel(labels));
ceCost = -sum(ceCost(I))/m;


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% err{nl} = zeros(10,1);
% for i = 1:numel(labels)
%     err{nl}(labels(i)) = err{nl}(labels(i))-1;
% end
% err{nl} = err{nl} + sum(pred_prob,2);
err{nl} = zeros(size(pred_prob));
err{nl}(I) = err{nl}(I) - 1;
err{nl} = err{nl} + pred_prob;


for l = nl-1:-1:2
    err{l} = transpose(stack{l}.W)*err{l+1} .* Act{l}.*(1-Act{l});
end


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for l = 1:nl-1
    gradStack{l}.W = err{l+1}*transpose(Act{l})/m + ei.lambda*stack{l}.W;
    gradStack{l}.b = sum(err{l+1},2)/m;
end

wCost = 0;
for l = 1:nl-1
    wCost = wCost + sum((stack{l}.W(:)).^2);
end
wCost = wCost * 0.5 * ei.lambda;
cost = ceCost + wCost;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end




