function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and
%   produces a feature vector from the word indices.

n = 1899;
x = zeros(n, 1);
x(word_indices) = 1;
end
