function [J grad] = nnCostFunction(nn_params, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, ...
X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));
m = size(X, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];

z3 = a2 * Theta2';
h = sigmoid(z3);

ident = eye(num_labels);
Y = ident(y,:);

J =1/m * sum(sum(-Y .* log(h)) - sum((1 - Y) .* log(1 - h)));
Theta1Ex = Theta1(:,2:end);
Theta2Ex = Theta2(:,2:end);
reg = lambda/(2*m) * (sum(sum(Theta1Ex.^2)) + sum(sum(Theta2Ex.^2)));

J = J + reg;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.


delta_2 = 0;
delta_1 = 0;

for t = 1:m
  a1 = [1; X(t,:)'];
  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  yt = Y(t,:)';
  d3 = a3 - yt;
  d2 = (Theta2Ex' * d3) .* sigmoidGradient(z2);

  delta_2 = delta_2 + d3 * a2';
  delta_1 = delta_1 + d2 * a1';
end

Theta1_grad = (1/m) * delta_1 + lambda/m * [zeros(size(Theta1, 1),1) Theta1(:,2:end)];
Theta2_grad = (1/m) * delta_2 + lambda/m * [zeros(size(Theta2, 1),1) Theta2(:,2:end)];

grad = [Theta1_grad(:); Theta2_grad(:);];

end
