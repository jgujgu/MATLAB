function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

C = 0;
sigma = 0;
steps = [0.01 0.03 0.1 0.3 1 3 10 30];
stepSize = length(steps);
smallestError = 10;

for i = 1:stepSize
  tempC = steps(i);
  for j = 1:stepSize
    tempSigma = steps(j);
    model = svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
    predictions = svmPredict(model, Xval);
    predError = mean(double(predictions ~= yval));
    if predError < smallestError
      smallestError = predError;
      C = tempC;
      sigma = tempSigma;
    end
  end
end

end
