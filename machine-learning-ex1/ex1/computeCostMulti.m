function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples
J = 1/(2*m) * sum(((X * theta) - y).^2);
end
