function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

err = (X * Theta' - Y).*R;
sqErr = err.^2;
X_reg = sum(sum(X.^2));
Theta_reg = sum(sum(Theta.^2));
J = 1/2 * sum(sum(sqErr.*R)) + lambda/2 * (X_reg + Theta_reg);

X_grad = err * Theta + lambda * X;
Theta_grad = err' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
