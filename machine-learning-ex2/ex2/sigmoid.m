function g = sigmoid(z)

g = zeros(size(z));
e = exp(1);
g = 1./(1 + e.^(-z));

end
