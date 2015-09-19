function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples
h = sigmoid(X * theta);

J = 1/m * sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) + lambda/(2 * m) * sum(theta(2:end).^2);
grad = ((X' * (h - y)) + (lambda * theta))/m;
grad(1) = grad(1) - ((lambda * theta(1))/m);

end
