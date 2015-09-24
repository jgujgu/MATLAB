function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
h = X * theta;
grad = zeros(size(theta));

reg = lambda/(2*m) * sum(theta(2:end,:).^2);
J = 1/(2*m) * sum((h - y).^2) + reg;

grad0 = 1/m * (h - y)' * X;
grad = (1/m * (h - y)' * X) + lambda/m * theta';
grad(1) = grad0(1);
grad = grad(:);

end
