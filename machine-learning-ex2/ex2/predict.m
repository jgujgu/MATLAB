function p = predict(theta, X)
p = sigmoid(X * theta) >= .5;
end
