function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  J_history(iter) = computeCost(X, y, theta);
  theta = theta - alpha/m * (((X * theta) - y)' * X)';
end

end
