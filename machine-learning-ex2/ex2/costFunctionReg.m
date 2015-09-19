function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
J = 0;
j = size(theta);

h = sigmoid(X * theta);
J = 1/m * (-y' * log(h) - (1 - y') * log(1 - h)) + lambda/(2 * m) * sum(theta(2:j).^2);
grad0 = 1/m * (h - y)' * X;
grad = (1/m * (h - y)' * X) + lambda/m * theta';
grad(1) = grad0(1);

end
